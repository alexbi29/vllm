// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// NOTICE: W12 lossless 12-bit BF16 weight GEMM for SM120 (mma.sync). The
// kernels (w12_gemm_kernel, decode_n, w12_decode_kernel) and the launcher are
// ported from SpecPilot (https://github.com/StargazerX0/SpecPilot, commit
// 78c42af, Apache-2.0), file
// settings/pro6000_c1_qwen38-27b_dflash2/files/python/sglang/kernels/ops/gemm/
// specpilot_d2_w12.py. Changes: SGLang/env-knob/Q-view/ablation code removed;
// split-K partials and tile counters live in min(T, P) slots; runtime
// activation epilogue (none / SiLU / GELU-tanh); proxy fence before a ring
// slot is released to the TMA producer; allocation-free escape scatter and an
// embedding row-gather kernel added; libtorch-stable host wrapper.
//
// Format (see vllm/model_executor/layers/w12_linear.py): a BF16 weight of
// kernel row n is coded against a per-row 16-binade exponent window with top
// E_n. Each weight is a Q byte [c0 m6 m5 m4 | s c3 c2 c1] plus a B nibble
// [m3..m0]; the kernel rebuilds w~ = s * 2^(c-15) * 1.m straight into mma A
// fragments and scales the fp32 accumulator by the exact power of two
// 2^(E_n-127). Weights outside the window (zeros, subnormals, inf/NaN and
// out-of-window exponents) decode as the placeholder 2^-15 and are corrected
// exactly from a per-row CSR list (acc += x*val - x*2^-15). Records are
// 128 rows x 64 k = 12 KB (8 KB Q region + 4 KB B region) in A-fragment lane
// order.

#include "torch_utils.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>

namespace {

using torch::headeronly::ScalarType;

constexpr int kNcw = 8;       // consumer warps (16 weight rows each)
constexpr int kRb = 128;      // weight rows per tile
constexpr int kWrec = 12288;  // bytes per (tile, 64-k) record
constexpr int kRot = 37;      // stage-order rotation within a segment
constexpr int kMaxM = 64;     // rows per launch
constexpr int kMaxDevices = 64;

__device__ __forceinline__ void mbar_init(uint32_t a, int cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(cnt));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t a, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(a),
      "r"(bytes)
      : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(a)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n W12W_%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra W12W_%=;\n}\n" ::"r"(a),
      "r"(parity)
      : "memory");
}
// The .shared::cta destination forms (no cluster launch). The .shared::cluster
// forms compile to a helper call on sm_120 that uses a stack frame, and any
// stack use makes the driver reserve local memory for every resident thread
// (3.7 GiB on a 188-SM RTX PRO 6000, taken from the KV cache).
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src,
                                         uint32_t bytes, uint32_t mbar) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::"
      "cache_hint [%0], [%1], %2, [%3], %4;\n" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(mbar), "l"(0x12F0000000000000ull)
      : "memory");
}
__device__ __forceinline__ void tma2d(uint32_t dst, const CUtensorMap* map,
                                      int c0, int c1, uint32_t mbar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(dst),
      "l"(map), "r"(c0), "r"(c1), "r"(mbar)
      : "memory");
}
__device__ __forceinline__ float ldg_early(const float* p) {
  float v;
  asm volatile("ld.global.nc.f32 %0, [%1];\n" : "=f"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ void cons_bar(int n) {
  asm volatile("bar.sync 1, %0;\n" ::"r"(n) : "memory");
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&b)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldsm_x2(uint32_t (&b)[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(b[0]), "=r"(b[1])
               : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* d, const uint32_t (&a)[4],
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Q words (2 uint4, one byte per weight) + B words (1 uint4 of m3..m0
// nibbles; word j low nibbles -> Q word 2j, high -> 2j+1) -> A fragments of
// 4 k16 steps; bf16 bits [s 0111 c3c2c1 | c0 m6..m0].
__device__ __forceinline__ void decode_n(const uint4 cq0, const uint4 cq1,
                                         const uint4 cb, uint32_t (&A)[4][4]) {
  const uint32_t Q[8] = {cq0.x, cq0.y, cq0.z, cq0.w,
                         cq1.x, cq1.y, cq1.z, cq1.w};
  const uint32_t B[4] = {cb.x, cb.y, cb.z, cb.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int c8 = 0; c8 < 2; ++c8) {
      const uint32_t q = Q[2 * j + c8];
      const uint32_t b = c8 ? (B[j] >> 4) & 0x0F0F0F0Fu : B[j] & 0x0F0F0F0Fu;
      const uint32_t lo = (q & 0xF0F0F0F0u) | b;
      const uint32_t hi =
          (((q << 4) & 0x80808080u) | (q & 0x07070707u)) | 0x38383838u;
      A[j][2 * c8] = __byte_perm(lo, hi, 0x5140);
      A[j][2 * c8 + 1] = __byte_perm(lo, hi, 0x7362);
    }
  }
}

__device__ __forceinline__ float act_mul(int act, float g, float u) {
  if (act == 1) return g / (1.f + __expf(-g)) * u;  // SiLU (one rounding)
  // GELU-tanh as vLLM's gelu_tanh_and_mul: act rounded to bf16, product
  // rounded by the caller.
  const float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
  const float a =
      __bfloat162float(__float2bfloat16(0.5f * g * (1.f + tanhf(inner))));
  return a * u;
}

// NT: token tile (M <= NT), NS: ring depth, NCB: correction buffers (2 = the
// escape warps run one segment ahead). Split-K partials / tile counters live
// in min(T, P) slots: indexed by tile when T <= P, by the tile's owner CTA
// (pfirst) when T > P (each CTA then owns at most one split tile).
template <int NT, int NS, int NCB>
__global__ void __launch_bounds__(kNcw * 32 + 96, 1)
    w12_gemm_kernel(const __grid_constant__ CUtensorMap xmap,
                    const __nv_bfloat16* __restrict__ x, int64_t ldx, int M,
                    const unsigned char* __restrict__ planes, int64_t tstride,
                    int S, int U, const float* __restrict__ scale,
                    const int* __restrict__ esc_rp,
                    const int* __restrict__ esc_col,
                    const float* __restrict__ esc_val,
                    __nv_bfloat16* __restrict__ out, int64_t ldo, int N,
                    float* __restrict__ ws, int maxseg,
                    int* __restrict__ counters, int act, int nout, int esc_seg,
                    int trig, int rot, int wsown) {
  constexpr int NQ = NT / 8;
  constexpr int XB = NT * 128;
  constexpr int RBY = kWrec;
  constexpr int SB = RBY + XB;  // multiple of 1024 (128B swizzle atom)
  constexpr int CB = kRb * NT;
  constexpr uint32_t TXB = SB;
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) uint64_t bars[2 * NS + 2 * NCB];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int P = gridDim.x, p = blockIdx.x;
  // 32-bit unit arithmetic avoids 64-bit division helper calls; the host
  // checks (P + 1) * U < 2^31.
  const int u0 = (int)((unsigned)(p * U) / (unsigned)P);
  const int u1 = (int)((unsigned)((p + 1) * U) / (unsigned)P);
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* const smem_al = smem + (sbase - sraw);
  float* corrbuf = reinterpret_cast<float*>(smem_al + NS * SB);
  float* xchg = corrbuf + NCB * CB;
  const uint32_t full0 = (uint32_t)__cvta_generic_to_shared(bars);
  const uint32_t empty0 = full0 + 8 * NS;
  const uint32_t cfull0 = full0 + 16 * NS, cempty0 = cfull0 + 8 * NCB;
  if (threadIdx.x == 0) {
    for (int s = 0; s < NS; ++s) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, kNcw);
    }
    for (int b = 0; b < NCB; ++b) {
      mbar_init(cfull0 + 8 * b, 64);
      mbar_init(cempty0 + 8 * b, kNcw * 32);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (trig) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  if (u0 >= u1) return;
  if (warp == kNcw) {  // producer
    if (lane == 0) {
      int tile = 0, sbeg = 0, slen = 1, seg_rot = 0;
      auto walk = [&](int u) -> int {
        if (u == u0 || u - tile * S == S) {
          tile = u / S;
          sbeg = u - tile * S;
          slen = min(u1, (tile + 1) * S) - u;
          seg_rot =
              rot ? (int)((unsigned)(tile * rot + p) % (unsigned)slen) : 0;
        }
        int s = (u - tile * S) - sbeg + seg_rot;
        if (s >= slen) s -= slen;
        return s + sbeg;
      };
      const int npre = min(NS, u1 - u0);
      int spre[NS];
#pragma unroll
      for (int i = 0; i < NS; ++i) {
        if (i < npre) {
          const int s = walk(u0 + i);
          spre[i] = s;
          const uint32_t fb = full0 + 8 * i;
          mbar_expect_tx(fb, TXB);
          bulk_g2s(sbase + i * SB,
                   planes + (int64_t)tile * tstride + (int64_t)s * kWrec, RBY,
                   fb);
        }
      }
      // PDL: the weight prologue above overlaps the predecessor's tail; x is
      // read only after this wait.
      asm volatile("griddepcontrol.wait;\n" ::: "memory");
#pragma unroll
      for (int i = 0; i < NS; ++i)
        if (i < npre)
          tma2d(sbase + i * SB + RBY, &xmap, spre[i] * 64, 0, full0 + 8 * i);
      for (int u = u0 + npre; u < u1; ++u) {
        const int i = u - u0, slot = i % NS;
        mbar_wait(empty0 + 8 * slot, ((i / NS) - 1) & 1);
        const int s = walk(u);
        const uint32_t fb = full0 + 8 * slot, dst = sbase + slot * SB;
        mbar_expect_tx(fb, TXB);
        bulk_g2s(dst, planes + (int64_t)tile * tstride + (int64_t)s * kWrec,
                 RBY, fb);
        tma2d(dst + RBY, &xmap, s * 64, 0, fb);
      }
    }
    return;
  }
  if (warp > kNcw) {  // 2 escape warps: exact correction of this segment
    const int ew = warp - kNcw - 1;
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    int j = 0;
    for (int u = u0; u < u1;) {
      const int tile = u / S, s0 = u - tile * S;
      const int uend = min(u1, (tile + 1) * S), s1 = uend - tile * S;
      if (!esc_seg && s0 != 0) {
        u = uend;
        continue;
      }
      const int k0 = esc_seg ? s0 * 64 : 0, k1 = esc_seg ? s1 * 64 : S * 64;
      const int b = j % NCB;
      if (j >= NCB) mbar_wait(cempty0 + 8 * b, ((j / NCB) - 1) & 1);
      float* cb = corrbuf + b * CB;
      for (int r = lane + 32 * ew; r < kRb; r += 64) {
        float c[NT];
#pragma unroll
        for (int m = 0; m < NT; ++m) c[m] = 0.f;
        const int n = tile * kRb + r;
        if (n < N) {
          const int e0 = esc_rp[n], e1 = esc_rp[n + 1];
          for (int e = e0; e < e1; ++e) {
            const int col = esc_col[e];
            if (col < k0 || col >= k1) continue;
            const float ev = esc_val[e];
            const __nv_bfloat16* xc = x + col;
#pragma unroll
            for (int m = 0; m < NT; ++m)
              if (m < M) {
                const float xv = __bfloat162float(xc[(int64_t)m * ldx]);
                // x * w~_true - x * 2^-15
                c[m] = fmaf(ev, xv, fmaf(-3.0517578125e-05f, xv, c[m]));
              }
          }
        }
#pragma unroll
        for (int m = 0; m < NT; ++m) cb[r * NT + ((m + r) % NT)] = c[m];
      }
      mbar_arrive(cfull0 + 8 * b);
      ++j;
      u = uend;
    }
    return;
  }
  // consumers: warp w owns weight rows [w*16, w*16+16) of the tile; tokens as
  // NQ n-tiles of 8.
  const int g = lane / 4, t = lane % 4;
  float acc[NQ * 4];
  // ldmatrix.x4 lane address: matrix mi = lane/8 (bit0: k half, bit1: n-tile
  // of the pair), row r = lane%8.
  const int lm_r = lane & 7, lm_kh = (lane >> 3) & 1, lm_nt = lane >> 4;
  auto stage = [&](int i) {
    const int slot = i % NS;
    mbar_wait(full0 + 8 * slot, (i / NS) & 1);
    const uint4* sw = reinterpret_cast<const uint4*>(smem_al + slot * SB);
    uint32_t A[4][4];
    const uint4 q0 = sw[warp * 64 + lane], q1 = sw[warp * 64 + 32 + lane];
    const uint4 bb = sw[512 + warp * 32 + lane];
    decode_n(q0, q1, bb, A);
    const uint32_t xs = sbase + slot * SB + RBY;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int qp = 0; qp < NQ / 2; ++qp) {
        uint32_t b[4];
        const int n = (2 * qp + lm_nt) * 8 + lm_r, ch = 2 * j + lm_kh;
        ldsm_x4(b, xs + n * 128 + ((ch ^ lm_r) << 4));
        mma16816(acc + (2 * qp) * 4, A[j], b[0], b[1]);
        mma16816(acc + (2 * qp + 1) * 4, A[j], b[2], b[3]);
      }
      if (NQ & 1) {
        uint32_t b[2];
        const int n = (NQ - 1) * 8 + lm_r, ch = 2 * j + lm_kh;
        ldsm_x2(b, xs + n * 128 + ((ch ^ lm_r) << 4));
        mma16816(acc + (NQ - 1) * 4, A[j], b[0], b[1]);
      }
    }
    __syncwarp();
    // The slot was read through the generic proxy (LDS, ldmatrix) and is
    // refilled through the async proxy (TMA); order the reads first.
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    if (lane == 0) mbar_arrive(empty0 + 8 * slot);
  };
  int jc = 0;
  for (int u = u0; u < u1;) {
    const int tile = u / S, s0 = u - tile * S;
    const int uend = min(u1, (tile + 1) * S);
    const int n0 = tile * kRb + warp * 16 + g;
    const float sc0 = n0 < N ? ldg_early(scale + n0) : 1.f;
    const float sc1 = n0 + 8 < N ? ldg_early(scale + n0 + 8) : 1.f;
#pragma unroll
    for (int q = 0; q < NQ * 4; ++q) acc[q] = 0.f;
    for (int uu = u; uu < uend; ++uu) stage(uu - u0);
    if (esc_seg || s0 == 0) {
      const int b = jc % NCB;
      const float* cb = corrbuf + b * CB;
      mbar_wait(cfull0 + 8 * b, (jc / NCB) & 1);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = warp * 16 + g + 8 * h;
#pragma unroll
        for (int q = 0; q < NQ; ++q)
#pragma unroll
          for (int v = 0; v < 2; ++v) {
            const int m = q * 8 + 2 * t + v;
            acc[q * 4 + 2 * h + v] += cb[r * NT + ((m + r) % NT)];
          }
      }
      mbar_arrive(cempty0 + 8 * b);
      ++jc;
    }
    const bool whole = (s0 == 0) && (uend == (tile + 1) * S);
    bool emit = whole;
    if (!whole) {
      const int ufirst = tile * S, ulast = ufirst + S - 1;
      const int pfirst = (int)((unsigned)((ufirst + 1) * P - 1) / (unsigned)U);
      const int plast = (int)((unsigned)((ulast + 1) * P - 1) / (unsigned)U);
      const int seg = p - pfirst, nseg = plast - pfirst + 1;
      const int widx = wsown ? pfirst : tile;
      float* wt = ws + ((int64_t)widx * maxseg) * kRb * NT;
      const int rl = warp * 16 + g;
      if (seg > 0) {
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int q = 0; q < NQ; ++q)
            __stcg(reinterpret_cast<float2*>(
                       wt + ((int64_t)seg * kRb + rl + 8 * h) * NT + q * 8 +
                       2 * t),
                   make_float2(acc[q * 4 + 2 * h], acc[q * 4 + 2 * h + 1]));
        cons_bar(kNcw * 32);
        if (threadIdx.x == 0)
          asm volatile(
              "red.release.gpu.global.add.s32 [%0], 1;\n" ::"l"(counters + widx)
              : "memory");
      } else {
        if (threadIdx.x == 0) {
          int c;
          do {
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n"
                         : "=r"(c)
                         : "l"(counters + widx)
                         : "memory");
          } while (c < nseg - 1);
          counters[widx] = 0;
        }
        cons_bar(kNcw * 32);
        constexpr int PB = NT <= 32 ? 4 : (NT <= 48 ? 2 : 1);
        for (int s2 = 1; s2 < nseg; s2 += PB) {
          float2 v[PB][2][NQ];
#pragma unroll
          for (int jj = 0; jj < PB; ++jj)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
              for (int q = 0; q < NQ; ++q)
                v[jj][h][q] =
                    s2 + jj < nseg
                        ? __ldcg(reinterpret_cast<const float2*>(
                              wt +
                              ((int64_t)(s2 + jj) * kRb + rl + 8 * h) * NT +
                              q * 8 + 2 * t))
                        : make_float2(0.f, 0.f);
#pragma unroll
          for (int jj = 0; jj < PB; ++jj)
            if (s2 + jj < nseg) {
#pragma unroll
              for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int q = 0; q < NQ; ++q) {
                  acc[q * 4 + 2 * h] += v[jj][h][q].x;
                  acc[q * 4 + 2 * h + 1] += v[jj][h][q].y;
                }
            }
        }
        emit = true;
      }
    }
    if (emit) {
      if (!act) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int n = n0 + 8 * h;
          const float sc = h ? sc1 : sc0;
#pragma unroll
          for (int q = 0; q < NQ; ++q)
#pragma unroll
            for (int v = 0; v < 2; ++v) {
              const int m = q * 8 + 2 * t + v;
              if (m < M && n < N)
                out[(int64_t)m * ldo + n] =
                    __float2bfloat16(acc[q * 4 + 2 * h + v] * sc);
            }
        }
      } else {
        // Tile rows = [64 gate | 64 up]: warps 4-7 (up) hand their
        // bf16-rounded rows to warps 0-3 (gate).
        const int wr = (warp & 3) * 16 + g;
        if (warp >= 4) {
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int q = 0; q < NQ; ++q)
#pragma unroll
              for (int v = 0; v < 2; ++v)
                xchg[(wr + 8 * h) * NT + q * 8 + 2 * t + v] = __bfloat162float(
                    __float2bfloat16(acc[q * 4 + 2 * h + v] * (h ? sc1 : sc0)));
        }
        cons_bar(kNcw * 32);
        if (warp < 4) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int n = tile * 64 + wr + 8 * h;
#pragma unroll
            for (int q = 0; q < NQ; ++q)
#pragma unroll
              for (int v = 0; v < 2; ++v) {
                const int m = q * 8 + 2 * t + v;
                const float gg = __bfloat162float(
                    __float2bfloat16(acc[q * 4 + 2 * h + v] * (h ? sc1 : sc0)));
                const float uu = xchg[(wr + 8 * h) * NT + q * 8 + 2 * t + v];
                if (m < M && n < nout)
                  out[(int64_t)m * ldo + n] =
                      __float2bfloat16(act_mul(act, gg, uu));
              }
          }
        }
        cons_bar(kNcw * 32);
      }
    }
    u = uend;
  }
}

// planes -> bf16 weight in the original row order (escapes are patched by
// w12_scatter); one block = one (tile, 64-k chunk).
__global__ void __launch_bounds__(256)
    w12_decode_kernel(const uint4* __restrict__ planes, int64_t tstride16,
                      int nkc, const float* __restrict__ scale,
                      __nv_bfloat16* __restrict__ out, int N, int K,
                      int interleave) {
  __shared__ uint32_t tl[128][33];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int kc = blockIdx.x % nkc, tile = blockIdx.x / nkc;
  const uint4* blk = planes + tile * tstride16 + (int64_t)kc * 768;
  uint32_t A[4][4];
  decode_n(blk[w * 64 + lane], blk[w * 64 + 32 + lane],
           blk[512 + w * 32 + lane], A);
  const int g = lane / 4, t = lane % 4;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int rl = w * 16 + g + 8 * h;
    const uint32_t E = (__float_as_uint(scale[tile * 128 + rl]) >> 23) & 0xFF;
    const uint32_t add = ((E - 127u) << 7) & 0xFFFFu;
    const uint32_t add2 = add | (add << 16);
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int c8 = 0; c8 < 2; ++c8)
        tl[rl][j * 8 + c8 * 4 + t] = __vadd2(A[j][2 * c8 + h], add2);
  }
  __syncthreads();
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int idx = threadIdx.x + 256 * q, rl = idx >> 3, c16 = idx & 7;
    const int r = tile * 128 + rl;
    int orig = r;
    if (interleave)
      orig = rl < 64 ? tile * 64 + rl : N / 2 + tile * 64 + rl - 64;
    if (r >= N || orig >= N) continue;
    const uint4 v = make_uint4(tl[rl][c16 * 4], tl[rl][c16 * 4 + 1],
                               tl[rl][c16 * 4 + 2], tl[rl][c16 * 4 + 3]);
    *reinterpret_cast<uint4*>(out + (int64_t)orig * K + kc * 64 + c16 * 8) = v;
  }
}

// Exact escape patch of a decoded weight: out[idx[i]] = val[i].
__global__ void w12_scatter_kernel(__nv_bfloat16* __restrict__ out,
                                   const int64_t* __restrict__ idx,
                                   const __nv_bfloat16* __restrict__ val,
                                   int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[idx[i]] = val[i];
}

// Embedding lookup from a plain (non-interleaved) W12 weight: out[t] =
// W[ids[t]] bit-exactly. One block per token; 4 threads per 64-k chunk decode
// the row's half of their A-fragment lanes, then the row's escapes are patched.
__global__ void __launch_bounds__(256)
    w12_embedding_kernel(const uint4* __restrict__ planes, int64_t tstride16,
                         int nkc, const float* __restrict__ scale,
                         const int* __restrict__ esc_rp,
                         const int* __restrict__ esc_col,
                         const __nv_bfloat16* __restrict__ esc_bf16,
                         const int64_t* __restrict__ ids,
                         __nv_bfloat16* __restrict__ out, int64_t ldo, int N) {
  const int64_t token = blockIdx.x;
  const int64_t n = ids[token];
  uint32_t* orow = reinterpret_cast<uint32_t*>(out + token * ldo);
  if (n < 0 || n >= N) {  // never produced by vLLM at TP=1; keep it defined
    for (int i = threadIdx.x; i < nkc * 32; i += blockDim.x) orow[i] = 0u;
    return;
  }
  const int tile = (int)(n / 128), rl = (int)(n % 128);
  const int w = rl / 16, g = rl % 8, h = (rl % 16) / 8;
  const uint32_t E = (__float_as_uint(scale[n]) >> 23) & 0xFF;
  const uint32_t add = ((E - 127u) << 7) & 0xFFFFu;
  const uint32_t add2 = add | (add << 16);
  for (int i = threadIdx.x; i < nkc * 4; i += blockDim.x) {
    const int kc = i >> 2, t = i & 3, lane = g * 4 + t;
    const uint4* blk = planes + tile * tstride16 + (int64_t)kc * 768;
    uint32_t A[4][4];
    decode_n(blk[w * 64 + lane], blk[w * 64 + 32 + lane],
             blk[512 + w * 32 + lane], A);
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int c8 = 0; c8 < 2; ++c8)
        orow[kc * 32 + j * 8 + c8 * 4 + t] = __vadd2(A[j][2 * c8 + h], add2);
  }
  __syncthreads();
  __nv_bfloat16* row = out + token * ldo;
  for (int e = esc_rp[n] + threadIdx.x; e < esc_rp[n + 1]; e += blockDim.x)
    row[esc_col[e]] = esc_bf16[e];
}

void check_cuda(cudaError_t err, const char* what) {
  STD_TORCH_CHECK(err == cudaSuccess, "W12 ", what, ": ",
                  cudaGetErrorString(err));
}

PFN_cuTensorMapEncodeTiled_v12000 encode_tiled() {
  static const PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult status;
    const cudaError_t err = cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &status);
    STD_TORCH_CHECK(err == cudaSuccess && status == cudaDriverEntryPointSuccess,
                    "W12: failed to resolve cuTensorMapEncodeTiled");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }();
  return fn;
}

template <int NT, int NS, int NCB>
constexpr int gemm_smem_bytes() {
  return NS * (kWrec + NT * 128) + NCB * kRb * NT * 4 + 64 * NT * 4 + 1024;
}

template <int NT, int NS, int NCB>
void launch_gemm(const torch::stable::Tensor& x,
                 const torch::stable::Tensor& planes,
                 const torch::stable::Tensor& scale,
                 const torch::stable::Tensor& rp,
                 const torch::stable::Tensor& col,
                 const torch::stable::Tensor& val, torch::stable::Tensor& out,
                 torch::stable::Tensor& ws, torch::stable::Tensor& cnt,
                 int64_t P, int64_t maxseg, int64_t N, int64_t act,
                 bool esc_seg, bool pdl, cudaStream_t stream) {
  constexpr int smem = gemm_smem_bytes<NT, NS, NCB>();
  static_assert(smem + 8 * (2 * NS + 2 * NCB) <= 101376,
                "W12 smem exceeds the SM120 per-block opt-in limit");
  const int M = (int)x.size(0), K = (int)x.size(1);
  const int nout = (int)out.size(1);
  const int T = (int)planes.size(0), S = K / 64, U = T * S;
  STD_TORCH_CHECK(P > 0 && P <= U, "W12: bad CTA count ", P);
  STD_TORCH_CHECK((P + 1) * (int64_t)U < (int64_t{1} << 31) &&
                      (int64_t)T * kRot + P < (int64_t{1} << 31),
                  "W12: weight too large for 32-bit stream-K unit indices");
  const bool wsown = T > P;
  const int64_t nslot = wsown ? P : T;
  STD_TORCH_CHECK(ws.numel() >= nslot * maxseg * kRb * NT,
                  "W12: split-K workspace too small");
  STD_TORCH_CHECK(cnt.numel() >= nslot, "W12: counter array too small");
  CUtensorMap map;
  cuuint64_t gdim[2] = {(cuuint64_t)K, (cuuint64_t)M};
  cuuint64_t gstr[1] = {(cuuint64_t)x.stride(0) * 2};
  cuuint32_t box[2] = {64, NT};
  cuuint32_t es[2] = {1, 1};
  const CUresult r = encode_tiled()(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
      const_cast<void*>(x.data_ptr()), gdim, gstr, box, es,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  STD_TORCH_CHECK(r == CUDA_SUCCESS, "W12: tensor map encode failed ", (int)r);
  auto kern = w12_gemm_kernel<NT, NS, NCB>;
  // Function attributes are per device; set them once per device.
  static std::atomic<uint64_t> attr_done{0};
  const int dev = x.get_device_index();
  const uint64_t bit = 1ull << (dev % kMaxDevices);
  if (!(attr_done.load(std::memory_order_acquire) & bit)) {
    check_cuda(cudaFuncSetAttribute(
                   kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
               "smem attribute");
    attr_done.fetch_or(bit, std::memory_order_acq_rel);
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)P);
  cfg.blockDim = dim3(kNcw * 32 + 96);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  check_cuda(
      cudaLaunchKernelEx(
          &cfg, kern, map, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
          (int64_t)x.stride(0), M,
          reinterpret_cast<const unsigned char*>(planes.data_ptr()),
          (int64_t)planes.stride(0), S, U,
          reinterpret_cast<const float*>(scale.data_ptr()),
          reinterpret_cast<const int*>(rp.data_ptr()),
          reinterpret_cast<const int*>(col.data_ptr()),
          reinterpret_cast<const float*>(val.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
          (int64_t)out.stride(0), (int)N,
          reinterpret_cast<float*>(ws.mutable_data_ptr()), (int)maxseg,
          reinterpret_cast<int*>(cnt.mutable_data_ptr()), (int)act, nout,
          esc_seg ? 1 : 0, pdl ? 1 : 0, kRot, wsown ? 1 : 0),
      "gemm launch");
}

bool same_device(const torch::stable::Tensor& a,
                 const torch::stable::Tensor& b) {
  return a.is_cuda() && b.is_cuda() &&
         a.get_device_index() == b.get_device_index();
}

void check_planes(const torch::stable::Tensor& planes,
                  const torch::stable::Tensor& scale, int64_t N, int64_t K) {
  STD_TORCH_CHECK(K > 0 && K % 64 == 0, "W12: K must be a multiple of 64");
  STD_TORCH_CHECK(planes.scalar_type() == ScalarType::Byte &&
                      planes.dim() == 2 && planes.is_contiguous() &&
                      planes.size(1) == (K / 64) * kWrec,
                  "W12: planes must be contiguous uint8 [T, K/64*12288]");
  const int64_t T = planes.size(0);
  STD_TORCH_CHECK(N > (T - 1) * kRb && N <= T * kRb,
                  "W12: N inconsistent with planes");
  STD_TORCH_CHECK(scale.scalar_type() == ScalarType::Float &&
                      scale.is_contiguous() && scale.numel() >= T * kRb,
                  "W12: scale must be fp32 [T*128]");
  STD_TORCH_CHECK(same_device(planes, scale), "W12: device mismatch");
}

}  // namespace

// y[M, nout] = x[M, K] @ W^T (act != 0: act(gate) * up of an interleaved
// gate_up weight), M <= 64.
void w12_gemm(torch::stable::Tensor const& x,
              torch::stable::Tensor const& planes,
              torch::stable::Tensor const& scale,
              torch::stable::Tensor const& esc_rp,
              torch::stable::Tensor const& esc_col,
              torch::stable::Tensor const& esc_val, torch::stable::Tensor& out,
              torch::stable::Tensor& workspace, torch::stable::Tensor& counters,
              int64_t num_ctas, int64_t max_segments, int64_t n, int64_t act,
              bool esc_per_segment, bool pdl) {
  STD_TORCH_CHECK(
      x.is_cuda() && x.scalar_type() == ScalarType::BFloat16 && x.dim() == 2,
      "W12: x must be CUDA bf16 [M, K]");
  const int64_t M = x.size(0), K = x.size(1);
  STD_TORCH_CHECK(M > 0 && M <= kMaxM, "W12: M must be in [1, 64], got ", M);
  STD_TORCH_CHECK(x.stride(1) == 1 && x.stride(0) % 8 == 0 &&
                      reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                  "W12: x rows must be unit-stride and 16-byte aligned");
  check_planes(planes, scale, n, K);
  const int64_t T = planes.size(0);
  STD_TORCH_CHECK(act >= 0 && act <= 2, "W12: act must be 0, 1 or 2");
  if (act) {
    STD_TORCH_CHECK(n % 128 == 0, "W12: fused act needs N % 128 == 0");
  }
  STD_TORCH_CHECK(esc_rp.scalar_type() == ScalarType::Int &&
                      esc_rp.numel() >= T * kRb + 1 && esc_rp.is_contiguous(),
                  "W12: esc_rp must be int32 [T*128+1]");
  STD_TORCH_CHECK(esc_col.scalar_type() == ScalarType::Int &&
                      esc_val.scalar_type() == ScalarType::Float &&
                      esc_col.numel() == esc_val.numel() &&
                      esc_col.is_contiguous() && esc_val.is_contiguous(),
                  "W12: escape CSR must be int32 col + fp32 val");
  STD_TORCH_CHECK(out.scalar_type() == ScalarType::BFloat16 && out.dim() == 2 &&
                      out.stride(1) == 1 && out.size(0) == M &&
                      out.size(1) == (act ? n / 2 : n),
                  "W12: out must be bf16 [M, nout] with unit last stride");
  STD_TORCH_CHECK(workspace.scalar_type() == ScalarType::Float &&
                      counters.scalar_type() == ScalarType::Int,
                  "W12: workspace must be fp32 and counters int32");
  const torch::stable::Tensor* operands[] = {&planes, &esc_rp, &out, &workspace,
                                             &counters};
  for (const torch::stable::Tensor* t : operands) {
    STD_TORCH_CHECK(same_device(x, *t), "W12: device mismatch");
  }
  if (esc_col.numel()) {
    STD_TORCH_CHECK(same_device(x, esc_col) && same_device(x, esc_val),
                    "W12: device mismatch");
  }
  const torch::stable::accelerator::DeviceGuard device_guard(
      x.get_device_index());
  cudaStream_t stream = get_current_cuda_stream(x.get_device_index());
#define W12_LAUNCH(NT_, NS_, NCB_)                                            \
  if (M <= NT_) {                                                             \
    launch_gemm<NT_, NS_, NCB_>(                                              \
        x, planes, scale, esc_rp, esc_col, esc_val, out, workspace, counters, \
        num_ctas, max_segments, n, act, esc_per_segment, pdl, stream);        \
    return;                                                                   \
  }
  // NT -> (ring depth, correction buffers), as measured by SpecPilot on
  // sm_120: 2-3 stages saturate GDDR7.
  W12_LAUNCH(8, 3, 2)
  W12_LAUNCH(16, 3, 2)
  W12_LAUNCH(24, 3, 2)
  W12_LAUNCH(32, 2, 2)
  W12_LAUNCH(40, 2, 2)
  W12_LAUNCH(48, 2, 2)
  W12_LAUNCH(64, 2, 1)
#undef W12_LAUNCH
}

// Decodes a W12 weight into bf16 [N, K] in the original row order; escape
// slots hold the placeholder until w12_scatter patches them.
void w12_decode(torch::stable::Tensor const& planes,
                torch::stable::Tensor const& scale, torch::stable::Tensor& out,
                int64_t n, int64_t k, bool interleave) {
  check_planes(planes, scale, n, k);
  STD_TORCH_CHECK(out.scalar_type() == ScalarType::BFloat16 &&
                      out.is_contiguous() && out.numel() >= n * k &&
                      same_device(planes, out),
                  "W12 decode: out must be contiguous bf16 with N*K elements");
  if (interleave) {
    STD_TORCH_CHECK(n % 128 == 0, "W12 decode: N % 128 != 0");
  }
  const torch::stable::accelerator::DeviceGuard device_guard(
      planes.get_device_index());
  const int T = (int)planes.size(0), nkc = (int)(k / 64);
  w12_decode_kernel<<<(unsigned)(T * nkc), 256, 0,
                      get_current_cuda_stream(planes.get_device_index())>>>(
      reinterpret_cast<const uint4*>(planes.data_ptr()), planes.stride(0) / 16,
      nkc, reinterpret_cast<const float*>(scale.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()), (int)n, (int)k,
      interleave ? 1 : 0);
  check_cuda(cudaGetLastError(), "decode launch");
}

void w12_scatter(torch::stable::Tensor& out, torch::stable::Tensor const& idx,
                 torch::stable::Tensor const& val) {
  const int64_t n = idx.numel();
  if (n == 0) return;
  STD_TORCH_CHECK(idx.scalar_type() == ScalarType::Long &&
                      val.scalar_type() == ScalarType::BFloat16 &&
                      out.scalar_type() == ScalarType::BFloat16 &&
                      val.numel() == n && idx.is_contiguous() &&
                      val.is_contiguous() && out.is_contiguous() &&
                      same_device(out, idx) && same_device(out, val),
                  "W12 scatter: bad arguments");
  const torch::stable::accelerator::DeviceGuard device_guard(
      out.get_device_index());
  w12_scatter_kernel<<<(unsigned)((n + 255) / 256), 256, 0,
                       get_current_cuda_stream(out.get_device_index())>>>(
      reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
      reinterpret_cast<const int64_t*>(idx.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(val.data_ptr()), n);
  check_cuda(cudaGetLastError(), "scatter launch");
}

// out[t, :] = W[ids[t], :] for a plain W12 weight (exact bf16 rows).
void w12_embedding(torch::stable::Tensor& out, torch::stable::Tensor const& ids,
                   torch::stable::Tensor const& planes,
                   torch::stable::Tensor const& scale,
                   torch::stable::Tensor const& esc_rp,
                   torch::stable::Tensor const& esc_col,
                   torch::stable::Tensor const& esc_bf16, int64_t n) {
  STD_TORCH_CHECK(ids.scalar_type() == ScalarType::Long && ids.dim() == 1 &&
                      ids.is_contiguous() && same_device(planes, ids),
                  "W12 embedding: ids must be contiguous int64 [T]");
  const int64_t ntok = ids.numel();
  STD_TORCH_CHECK(out.scalar_type() == ScalarType::BFloat16 && out.dim() == 2 &&
                      out.size(0) == ntok && out.stride(1) == 1 &&
                      out.stride(0) % 2 == 0 && same_device(planes, out),
                  "W12 embedding: out must be bf16 [T, K]");
  const int64_t k = out.size(1);
  check_planes(planes, scale, n, k);
  STD_TORCH_CHECK(esc_rp.scalar_type() == ScalarType::Int &&
                      esc_rp.numel() >= n + 1 && esc_rp.is_contiguous() &&
                      esc_col.scalar_type() == ScalarType::Int &&
                      esc_bf16.scalar_type() == ScalarType::BFloat16 &&
                      esc_col.numel() == esc_bf16.numel(),
                  "W12 embedding: bad escape arrays");
  if (ntok == 0) return;
  const torch::stable::accelerator::DeviceGuard device_guard(
      planes.get_device_index());
  const int nkc = (int)(k / 64);
  const int threads = nkc * 4 < 256 ? ((nkc * 4 + 31) / 32) * 32 : 256;
  w12_embedding_kernel<<<(unsigned)ntok, threads, 0,
                         get_current_cuda_stream(planes.get_device_index())>>>(
      reinterpret_cast<const uint4*>(planes.data_ptr()), planes.stride(0) / 16,
      nkc, reinterpret_cast<const float*>(scale.data_ptr()),
      reinterpret_cast<const int*>(esc_rp.data_ptr()),
      reinterpret_cast<const int*>(esc_col.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(esc_bf16.data_ptr()),
      reinterpret_cast<const int64_t*>(ids.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()), out.stride(0),
      (int)n);
  check_cuda(cudaGetLastError(), "embedding launch");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("w12_gemm", TORCH_BOX(&w12_gemm));
  m.impl("w12_decode", TORCH_BOX(&w12_decode));
  m.impl("w12_scatter", TORCH_BOX(&w12_scatter));
  m.impl("w12_embedding", TORCH_BOX(&w12_embedding));
}
