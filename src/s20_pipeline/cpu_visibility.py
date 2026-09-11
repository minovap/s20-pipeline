"""ctypes bridge to the exact mixed-precision CPU visibility kernel."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VisibilityResult:
    flags: np.ndarray
    incidence: np.ndarray
    denominator_shortcuts: int
    plane_shortcuts: int


class CpuVisibility:
    """Borrow immutable geometry and run stateless, allocation-free native chunks."""

    def __init__(self, xyz, normals, library: Path):
        self.xyz = np.asarray(xyz)
        self.normals = np.asarray(normals)
        if (
            self.xyz.dtype != np.float32
            or self.normals.dtype != np.float32
            or not self.xyz.flags.c_contiguous
            or not self.normals.flags.c_contiguous
            or self.xyz.shape != self.normals.shape
            or self.xyz.ndim != 2
            or self.xyz.shape[1] != 3
        ):
            raise ValueError("Native visibility expects contiguous matching n x 3 float32 arrays")
        if not len(self.xyz) or len(self.xyz) > np.iinfo(np.uint32).max:
            raise ValueError("Native visibility point count must fit uint32")
        self.library = ctypes.CDLL(str(library))
        self._configure()
        if self.library.s20_visibility_abi_version() != 1:
            raise RuntimeError("Unsupported native visibility ABI")
    def _configure(self):
        float_pointer = ctypes.POINTER(ctypes.c_float)
        double_pointer = ctypes.POINTER(ctypes.c_double)
        byte_pointer = ctypes.POINTER(ctypes.c_uint8)
        uint_pointer = ctypes.POINTER(ctypes.c_uint32)
        ulong_pointer = ctypes.POINTER(ctypes.c_uint64)
        self.library.s20_visibility_abi_version.argtypes = []
        self.library.s20_visibility_abi_version.restype = ctypes.c_uint32
        self.library.s20_visibility_decide.argtypes = [
            float_pointer,
            float_pointer,
            ctypes.c_uint64,
            uint_pointer,
            float_pointer,
            float_pointer,
            float_pointer,
            ulong_pointer,
            ulong_pointer,
            ctypes.c_uint64,
            float_pointer,
            double_pointer,
            byte_pointer,
            ctypes.c_uint32,
            ctypes.c_uint32,
            byte_pointer,
            float_pointer,
            ulong_pointer,
            ulong_pointer,
        ]
        self.library.s20_visibility_decide.restype = ctypes.c_int

    def decide(self, ids, u, v, distance, blocker_keys, exact_keys, frame, mask):
        ids = np.asarray(ids)
        if ids.ndim != 1:
            raise ValueError("Native visibility point IDs must be one-dimensional")
        if ids.dtype != np.uint32:
            if not np.issubdtype(ids.dtype, np.integer) or (
                len(ids) and (ids.min() < 0 or ids.max() > np.iinfo(np.uint32).max)
            ):
                raise ValueError("Native visibility point IDs must fit uint32")
            ids = ids.astype(np.uint32)
        ids = np.ascontiguousarray(ids)
        vectors = [np.asarray(value) for value in (u, v, distance, blocker_keys, exact_keys)]
        if any(value.ndim != 1 for value in vectors):
            raise ValueError("Native visibility inputs must be one-dimensional")
        u = np.ascontiguousarray(u, dtype=np.float32)
        v = np.ascontiguousarray(v, dtype=np.float32)
        distance = np.ascontiguousarray(distance, dtype=np.float32)
        blocker_keys = np.ascontiguousarray(blocker_keys)
        exact_keys = np.ascontiguousarray(exact_keys)
        if blocker_keys.dtype not in (np.dtype(np.int64), np.dtype(np.uint64)):
            blocker_keys = blocker_keys.astype(np.uint64)
        if exact_keys.dtype not in (np.dtype(np.int64), np.dtype(np.uint64)):
            exact_keys = exact_keys.astype(np.uint64)
        mask = self.prepare_mask(mask)
        count = len(ids)
        if any(len(value) != count for value in (u, v, distance, blocker_keys, exact_keys)):
            raise ValueError("Native visibility inputs must have equal lengths")
        center32 = np.ascontiguousarray(frame.center, dtype=np.float32)
        center64 = np.ascontiguousarray(frame.center, dtype=np.float64)
        if center32.shape != (3,):
            raise ValueError("Native visibility camera center must contain three values")
        flags = np.empty(count, dtype=np.uint8)
        incidence = np.empty(count, dtype=np.float32)
        denominator_shortcuts = ctypes.c_uint64()
        plane_shortcuts = ctypes.c_uint64()
        result = self.library.s20_visibility_decide(
            self.xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.normals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(self.xyz),
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            u.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            distance.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            blocker_keys.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            exact_keys.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            count,
            center32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            center64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            mask.shape[1],
            mask.shape[0],
            flags.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            incidence.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(denominator_shortcuts),
            ctypes.byref(plane_shortcuts),
        )
        if result:
            raise RuntimeError(f"Native visibility failed with error {result}")
        return VisibilityResult(
            flags,
            incidence,
            denominator_shortcuts.value,
            plane_shortcuts.value,
        )

    @staticmethod
    def prepare_mask(mask):
        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError("Native visibility mask must be two-dimensional")
        return np.ascontiguousarray(
            mask if mask.dtype in (np.dtype(np.bool_), np.dtype(np.uint8)) else mask != 0,
            dtype=np.uint8,
        )

    def stats(self):
        return {
            "backend": "C++ exact mixed-precision visibility and mask",
        }
