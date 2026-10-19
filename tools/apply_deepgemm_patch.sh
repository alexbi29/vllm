#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

src_dir=${1:?Usage: apply_deepgemm_patch.sh DEEPGEMM_SRC_DIR}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
patch_file="$script_dir/../cmake/patches/deepgemm-sm120-fp8-fp4-m32.patch"

# Reconfiguration and already-patched local sources must be idempotent.
if git -C "$src_dir" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    exit 0
fi

# Refuse incompatible or partially patched sources before making any changes.
git -C "$src_dir" apply --check "$patch_file"
git -C "$src_dir" apply "$patch_file"
