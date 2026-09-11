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
        self.library.s20_visibility_abi_version.argtypes = []
        self.library.s20_visibility_abi_version.restype = ctypes.c_uint32
        if self.library.s20_visibility_abi_version() != 3:
            raise RuntimeError("Unsupported native visibility ABI")
        self._configure()
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
        self.library.s20_rank_insert.argtypes = [
            float_pointer,
            ctypes.c_uint64,
            uint_pointer,
            float_pointer,
            float_pointer,
            float_pointer,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ulong_pointer,
        ]
        self.library.s20_rank_insert.restype = ctypes.c_int
        self.library.s20_bucket_slots.argtypes = [
            float_pointer,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ulong_pointer,
            ulong_pointer,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        self.library.s20_bucket_slots.restype = ctypes.c_int

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

    def insert(self, observations, ids, u, v, scores, photo):
        """Insert one photo's candidates; ids must be unique within the call."""
        observations = np.asarray(observations)
        if (
            observations.dtype != np.float32
            or observations.shape != (len(self.xyz), 4, 8)
            or not observations.flags.c_contiguous
            or not observations.flags.writeable
        ):
            raise ValueError("Native ranking expects writable contiguous n x 4 x 8 float32 records")
        ids = np.asarray(ids)
        if ids.ndim != 1 or ids.dtype != np.uint32 or not ids.flags.c_contiguous:
            raise ValueError("Native ranking point IDs must be contiguous uint32")
        u = np.ascontiguousarray(u, dtype=np.float32)
        v = np.ascontiguousarray(v, dtype=np.float32)
        scores = np.ascontiguousarray(scores, dtype=np.float32)
        if any(value.ndim != 1 or len(value) != len(ids) for value in (u, v, scores)):
            raise ValueError("Native ranking inputs must be equal-length vectors")
        if not 0 <= photo <= np.iinfo(np.uint32).max:
            raise ValueError("Native ranking photo ID must fit uint32")
        inserted = ctypes.c_uint64()
        result = self.library.s20_rank_insert(
            observations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(self.xyz),
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            u.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(ids),
            photo,
            ctypes.byref(inserted),
        )
        if result:
            raise RuntimeError(f"Native ranking failed with error {result}")
        return inserted.value

    def bucket_slots(self, observations, offsets, slot_ids):
        observations = np.asarray(observations)
        offsets = np.ascontiguousarray(offsets, dtype=np.uint64)
        slot_ids = np.asarray(slot_ids)
        if (
            observations.dtype != np.float32
            or observations.ndim != 3
            or observations.shape[1:] != (4, 8)
            or not observations.flags.c_contiguous
        ):
            raise ValueError("Native buckets expect contiguous n x 4 x 8 float32 records")
        if offsets.ndim != 1 or len(offsets) < 2:
            raise ValueError("Native bucket offsets must contain every photo boundary")
        if (
            offsets[0] != 0
            or np.any(offsets[1:] < offsets[:-1])
            or int(offsets[-1]) != slot_ids.size
        ):
            raise ValueError("Native bucket offsets must exactly bound the slot index")
        if slot_ids.dtype not in (np.dtype(np.uint32), np.dtype(np.uint64)):
            raise ValueError("Native bucket slot IDs must be uint32 or uint64")
        if slot_ids.ndim != 1 or not slot_ids.flags.c_contiguous or not slot_ids.flags.writeable:
            raise ValueError("Native bucket slot IDs must be writable and contiguous")
        cursors = offsets[:-1].copy()
        result = self.library.s20_bucket_slots(
            observations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(observations),
            len(offsets) - 1,
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            cursors.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_void_p(slot_ids.ctypes.data),
            slot_ids.dtype.itemsize * 8,
        )
        if result:
            raise RuntimeError(f"Native bucket scatter failed with error {result}")
        if not np.array_equal(cursors, offsets[1:]):
            raise RuntimeError("Native candidate photo buckets are incomplete")

    def release_geometry(self):
        self.xyz = None
        self.normals = None

    def stats(self):
        return {
            "backend": "C++ exact mixed-precision visibility and mask",
        }
