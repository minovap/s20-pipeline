"""Persistent ctypes bridge to the native Metal collector kernels."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MetalPhotoResult:
    projection: np.ndarray
    depth_keys: np.ndarray
    exact_keys: np.ndarray
    blocker_keys: np.ndarray
    flags: np.ndarray
    projection_rechecks: int
    visibility_rechecks: int
    exact_depth_rechecks: int


class MetalCollector:
    """Own geometry buffers for a complete candidates stage."""

    def __init__(self, xyz, normals, library: Path, kernels: Path, chunk=262144):
        self.xyz = np.ascontiguousarray(xyz, dtype=np.float32)
        self.normals = np.ascontiguousarray(normals, dtype=np.float32)
        self.chunk = chunk
        if self.xyz.shape != self.normals.shape or self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError("Metal collector expects matching n x 3 geometry and normals")
        if len(self.xyz) > np.iinfo(np.uint32).max:
            raise ValueError("Metal collector point count exceeds uint32 indexing")
        if len(self.xyz):
            coordinate_bounds = np.concatenate((self.xyz.min(axis=0), self.xyz.max(axis=0)))
            self.coordinate_spacing = float(
                np.max(np.abs(np.spacing(coordinate_bounds.astype(np.float32))))
            )
        else:
            self.coordinate_spacing = 0.0
        self.library = ctypes.CDLL(str(library))
        self._configure()
        error = ctypes.create_string_buffer(4096)
        self.handle = self.library.s20_collector_create(
            self.xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.normals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(self.xyz),
            str(kernels).encode(),
            error,
            len(error),
        )
        if not self.handle:
            raise RuntimeError(error.value.decode(errors="replace"))

    def _configure(self):
        lib = self.library
        float_pointer = ctypes.POINTER(ctypes.c_float)
        uint_pointer = ctypes.POINTER(ctypes.c_uint32)
        lib.s20_collector_create.argtypes = [
            float_pointer,
            float_pointer,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.s20_collector_create.restype = ctypes.c_void_p
        lib.s20_collector_destroy.argtypes = [ctypes.c_void_p]
        lib.s20_collector_project.argtypes = [
            ctypes.c_void_p,
            uint_pointer,
            ctypes.c_uint32,
            float_pointer,
            float_pointer,
            float_pointer,
            float_pointer,
            float_pointer,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_float,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.s20_collector_project.restype = ctypes.c_int
        lib.s20_collector_finish.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.s20_collector_finish.restype = ctypes.c_int
        for name in (
            "s20_collector_projection",
            "s20_collector_depth_keys",
            "s20_collector_exact_keys",
            "s20_collector_blocker_keys",
            "s20_collector_flags",
        ):
            function = getattr(lib, name)
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_void_p
        lib.s20_collector_gpu_seconds.argtypes = [ctypes.c_void_p]
        lib.s20_collector_gpu_seconds.restype = ctypes.c_double
        lib.s20_collector_allocated_bytes.argtypes = [ctypes.c_void_p]
        lib.s20_collector_allocated_bytes.restype = ctypes.c_uint64
        lib.s20_collector_launches.argtypes = [ctypes.c_void_p]
        lib.s20_collector_launches.restype = ctypes.c_uint32
        lib.s20_collector_device_name.argtypes = [ctypes.c_void_p]
        lib.s20_collector_device_name.restype = ctypes.c_char_p

    def process(self, selected, frame) -> MetalPhotoResult:
        selected = np.ascontiguousarray(selected, dtype=np.uint32)
        center = np.ascontiguousarray(frame.center, dtype=np.float32)
        center_error = np.nextafter(
            np.abs(frame.center - center.astype(np.float64)).astype(np.float32),
            np.float32(np.inf),
        )
        rotation = np.ascontiguousarray(frame.camera_to_world, dtype=np.float32)
        coefficients = np.ascontiguousarray(frame.calibration.coefficients, dtype=np.float32)
        calibration = frame.calibration
        intrinsics = np.asarray(
            [calibration.a11, calibration.a12, calibration.a22, calibration.u0, calibration.v0],
            dtype=np.float32,
        )
        error = ctypes.create_string_buffer(4096)
        result = self.library.s20_collector_project(
            self.handle,
            selected.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            len(selected),
            center.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            center_error.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rotation.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            coefficients.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            intrinsics.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            calibration.width,
            calibration.height,
            np.float32(np.deg2rad(calibration.max_incident_angle_deg)),
            error,
            len(error),
        )
        if result:
            raise RuntimeError(error.value.decode(errors="replace"))
        count = len(selected)
        projection = self._array("s20_collector_projection", np.float32, count * 4).reshape(
            count, 4
        )
        depth_keys = self._array("s20_collector_depth_keys", np.uint64, count)
        flags = self._array("s20_collector_flags", np.uint32, count)
        projection_rechecks = self._preserve_cpu_projection_contract(
            selected, frame, projection, depth_keys, flags
        )
        result = self.library.s20_collector_finish(
            self.handle, calibration.width, calibration.height, error, len(error)
        )
        if result:
            raise RuntimeError(error.value.decode(errors="replace"))
        exact_keys = self._array("s20_collector_exact_keys", np.uint64, count)
        blocker_keys = self._array("s20_collector_blocker_keys", np.uint64, count)
        visibility_rechecks = self._preserve_mixed_visibility_contract(
            selected, frame, projection, exact_keys, blocker_keys, flags
        )
        return MetalPhotoResult(
            projection,
            depth_keys,
            exact_keys,
            blocker_keys,
            flags,
            projection_rechecks,
            visibility_rechecks,
            int(np.count_nonzero((flags & np.uint32(1)) != 0)),
        )

    def _preserve_cpu_projection_contract(self, selected, frame, projection, depth_keys, flags):
        """Keep CPU exact depths and recheck only GPU projection boundary cases."""
        from .camera import project

        count = len(selected)
        if count == 0:
            return 0
        center = frame.center.astype(np.float32)
        rotation = frame.camera_to_world.astype(np.float32)
        calibration = frame.calibration
        gpu_valid = (flags & np.uint32(1)) != 0
        positions = np.flatnonzero((flags & np.uint32(32)) != 0)
        for start in range(0, len(positions), self.chunk):
            positions_chunk = positions[start : start + self.chunk]
            cpu_projection = np.column_stack(project(self.xyz[selected[positions_chunk]], frame))
            projection[positions_chunk] = cpu_projection
        u, v, angle, distance = projection.T
        valid = gpu_valid.copy()
        if len(positions):
            valid[positions] = (
                (distance[positions] > 0.1)
                & np.isfinite(u[positions])
                & np.isfinite(v[positions])
                & (u[positions] >= 0)
                & (v[positions] >= 0)
                & (u[positions] < calibration.width - 1)
                & (v[positions] < calibration.height - 1)
                & (angle[positions] < np.deg2rad(calibration.max_incident_angle_deg))
            )

        repaired = np.zeros(count, dtype=bool)
        repaired[positions] = True
        valid_positions = np.flatnonzero(valid & ~repaired)
        for start in range(0, len(valid_positions), self.chunk):
            positions_chunk = valid_positions[start : start + self.chunk]
            camera = (self.xyz[selected[positions_chunk]] - center) @ rotation
            projection[positions_chunk, 3] = np.linalg.norm(camera, axis=1)
            radial = np.hypot(camera[:, 0], camera[:, 1])
            # Candidate scoring is CPU-owned and very close scores can change the
            # stable top-four order. Preserve the CPU incident angle exactly.
            projection[positions_chunk, 2] = np.arctan2(radial, camera[:, 2])
        u, v, angle, distance = projection.T
        valid = (
            valid
            & (distance > 0.1)
            & np.isfinite(u)
            & np.isfinite(v)
            & (u >= 0)
            & (v >= 0)
            & (u < calibration.width - 1)
            & (v < calibration.height - 1)
            & (angle < np.deg2rad(calibration.max_incident_angle_deg))
        )
        flags[:] = valid.astype(np.uint32)
        maximum_distance = (np.iinfo(np.int64).max - len(self.xyz)) // len(self.xyz) / 1e6
        if valid.any() and float(distance[valid].max()) >= maximum_distance:
            raise ValueError("Scene exceeds int64 depth encoding range")
        return int(len(positions))

    def _preserve_mixed_visibility_contract(
        self, selected, frame, projection, exact_keys, blocker_keys, flags
    ):
        """CPU-recheck only decisions close enough to a visibility threshold to flip."""
        from .collect import RELIABLE, SURFACE_REJECTED, VALID, VISIBLE, visibility_decisions

        if not len(selected):
            return 0
        valid = (flags & VALID) != 0
        ambiguous = valid & ((flags & np.uint32(16)) != 0)
        center_rounding = np.linalg.norm(frame.center - frame.center.astype(np.float32))
        center_spacing = float(
            np.max(np.abs(np.spacing(frame.center.astype(np.float32))), initial=0)
        )
        if center_rounding > 1e-4 or max(center_spacing, self.coordinate_spacing) > 1e-4:
            ambiguous = valid
        positions = np.flatnonzero(ambiguous)
        if not len(positions):
            flags[:] &= np.uint32(15)
            return 0
        for start in range(0, len(positions), self.chunk):
            positions_chunk = positions[start : start + self.chunk]
            blockers = (blocker_keys[positions_chunk] % np.uint64(len(self.xyz))).astype(np.int64)
            visible, reliable, rejected, _ = visibility_decisions(
                self.xyz,
                self.normals,
                selected[positions_chunk],
                projection[positions_chunk, 3],
                blockers,
                exact_keys[positions_chunk],
                len(self.xyz),
                frame,
                "mixed",
            )
            replacement = np.full(len(positions_chunk), VALID, dtype=np.uint32)
            replacement[reliable] |= RELIABLE
            replacement[rejected] |= SURFACE_REJECTED
            replacement[visible] |= VISIBLE
            flags[positions_chunk] = replacement
        flags[:] &= np.uint32(15)
        return int(len(positions))

    def _array(self, function, dtype, count):
        address = getattr(self.library, function)(self.handle)
        ctype = np.ctypeslib.as_ctypes_type(np.dtype(dtype))
        return np.ctypeslib.as_array((ctype * count).from_address(address))

    def stats(self):
        return {
            "device": self.library.s20_collector_device_name(self.handle).decode(),
            "gpu_command_s": self.library.s20_collector_gpu_seconds(self.handle),
            "allocated_buffer_bytes": self.library.s20_collector_allocated_bytes(self.handle),
            "launches": self.library.s20_collector_launches(self.handle),
            "fast_math": False,
            "precision": "Metal safe math with CPU exact-depth and ambiguous-boundary rechecks",
        }

    def close(self):
        if getattr(self, "handle", None):
            self.library.s20_collector_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()
