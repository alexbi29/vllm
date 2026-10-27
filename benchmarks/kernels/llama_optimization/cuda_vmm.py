# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark-only CUDA VMM buffers with verified compression properties."""

import ctypes as c


class Location(c.Structure):
    _fields_ = [("type", c.c_int), ("id", c.c_int)]


class AllocationFlags(c.Structure):
    _fields_ = [
        ("compression", c.c_ubyte),
        ("rdma", c.c_ubyte),
        ("usage", c.c_ushort),
        ("reserved", c.c_ubyte * 4),
    ]


class AllocationProperties(c.Structure):
    _fields_ = [
        ("type", c.c_int),
        ("handle_types", c.c_int),
        ("location", Location),
        ("win32", c.c_void_p),
        ("flags", AllocationFlags),
    ]


class Access(c.Structure):
    _fields_ = [("location", Location), ("flags", c.c_int)]


class VMMBuffer:
    """Own a mapped allocation in the caller's current CUDA context.

    Keep this object alive while tensors or graphs reference its memory. Closing
    synchronizes the probe context before unmapping; it is not an inference path.
    """

    def __init__(self, shape: tuple[int, ...], compressed: bool, device: int = 0):
        self.driver = c.CDLL("libcuda.so.1")
        u64, size = c.c_uint64, c.c_size_t
        signatures = {
            "cuCtxGetCurrent": [c.POINTER(c.c_void_p)],
            "cuCtxSynchronize": [],
            "cuDeviceGetAttribute": [c.POINTER(c.c_int), c.c_int, c.c_int],
            "cuMemGetAllocationGranularity": [
                c.POINTER(size),
                c.POINTER(AllocationProperties),
                c.c_int,
            ],
            "cuMemCreate": [c.POINTER(u64), size, c.POINTER(AllocationProperties), u64],
            "cuMemGetAllocationPropertiesFromHandle": [
                c.POINTER(AllocationProperties),
                u64,
            ],
            "cuMemAddressReserve": [c.POINTER(u64), size, size, u64, u64],
            "cuMemMap": [u64, size, size, u64, u64],
            "cuMemSetAccess": [u64, size, c.POINTER(Access), size],
            "cuMemUnmap": [u64, size],
            "cuMemRelease": [u64],
            "cuMemAddressFree": [u64, size],
        }
        for name, arguments in signatures.items():
            function = getattr(self.driver, name)
            function.argtypes = arguments
            function.restype = c.c_int
        assert c.sizeof(AllocationProperties) == 32 and c.sizeof(Access) == 12
        context = c.c_void_p()
        self.call("cuCtxGetCurrent", c.byref(context))
        if not context.value:
            raise RuntimeError("Initialize the PyTorch CUDA context before allocating")
        properties = AllocationProperties(type=1, location=Location(1, device))
        properties.flags.compression = int(compressed)
        granularity = size()
        self.call(
            "cuMemGetAllocationGranularity",
            c.byref(granularity),
            c.byref(properties),
            0,
        )
        count = 1
        for dimension in shape:
            count *= dimension
        self.allocated_bytes = (
            (count + granularity.value - 1) // granularity.value * granularity.value
        )
        self.pointer, self.handle = u64(), u64()
        self.mapped = False
        try:
            self.call(
                "cuMemCreate",
                c.byref(self.handle),
                self.allocated_bytes,
                c.byref(properties),
                0,
            )
            actual = AllocationProperties()
            self.call(
                "cuMemGetAllocationPropertiesFromHandle", c.byref(actual), self.handle
            )
            if actual.flags.compression != int(compressed):
                raise RuntimeError(
                    "Driver did not grant requested compression property"
                )
            self.call(
                "cuMemAddressReserve",
                c.byref(self.pointer),
                self.allocated_bytes,
                0,
                0,
                0,
            )
            self.call("cuMemMap", self.pointer, self.allocated_bytes, 0, self.handle, 0)
            self.mapped = True
            access = Access(Location(1, device), 3)
            self.call(
                "cuMemSetAccess", self.pointer, self.allocated_bytes, c.byref(access), 1
            )
            self.__cuda_array_interface__ = {
                "shape": shape,
                "strides": None,
                "typestr": "|u1",
                "data": (self.pointer.value, False),
                "version": 3,
            }
            self.metadata = {
                "requested_compression": int(compressed),
                "granted_compression": int(actual.flags.compression),
                "requested_bytes": count,
                "allocated_bytes": self.allocated_bytes,
                "granularity": granularity.value,
            }
        except BaseException:
            self.close()
            raise

    def call(self, name, *arguments):
        result = getattr(self.driver, name)(*arguments)
        if result:
            raise RuntimeError(f"{name} returned CUDA error {result}")

    def close(self):
        if self.mapped:
            self.call("cuCtxSynchronize")
            self.call("cuMemUnmap", self.pointer, self.allocated_bytes)
            self.mapped = False
        if self.pointer.value:
            self.call("cuMemAddressFree", self.pointer, self.allocated_bytes)
            self.pointer.value = 0
        if self.handle.value:
            self.call("cuMemRelease", self.handle)
            self.handle.value = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
