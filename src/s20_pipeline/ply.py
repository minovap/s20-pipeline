#!/usr/bin/env python3
"""Lossless binary little-endian PLY I/O for the native colorizer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "float": "<f4",
    "double": "<f8",
    "int8": "i1",
    "uint8": "u1",
    "int16": "<i2",
    "uint16": "<u2",
    "int32": "<i4",
    "uint32": "<u4",
    "float32": "<f4",
    "float64": "<f8",
}


@dataclass(frozen=True)
class PlyInfo:
    path: Path
    header_lines: tuple[bytes, ...]
    point_offset: int
    point_count: int
    point_length: int
    dtype: np.dtype


def read_ply_info(path: Path) -> PlyInfo:
    path = path.resolve()
    header_lines: list[bytes] = []
    properties: list[tuple[str, str]] = []
    point_count: int | None = None
    current_element: str | None = None
    non_vertex_elements: list[tuple[str, int]] = []
    format_seen = False

    with path.open("rb") as stream:
        first = stream.readline()
        if first.strip() != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        header_lines.append(first)
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError(f"PLY header ended unexpectedly: {path}")
            header_lines.append(raw)
            text = raw.decode("ascii").strip()
            if text == "format binary_little_endian 1.0":
                format_seen = True
            elif text.startswith("element "):
                _, current_element, count_text = text.split()
                count = int(count_text)
                if current_element == "vertex":
                    if point_count is not None:
                        raise ValueError(f"multiple vertex elements are unsupported: {path}")
                    point_count = count
                elif count:
                    non_vertex_elements.append((current_element, count))
            elif text.startswith("property ") and current_element == "vertex":
                words = text.split()
                if len(words) != 3 or words[1] == "list":
                    raise ValueError(f"list-valued vertex properties are unsupported: {path}")
                type_name, name = words[1], words[2]
                if type_name not in PLY_TYPES:
                    raise ValueError(f"unsupported PLY property type {type_name!r}: {path}")
                properties.append((name, PLY_TYPES[type_name]))
            elif text == "end_header":
                point_offset = stream.tell()
                break

    if not format_seen:
        raise ValueError(f"expected binary_little_endian PLY: {path}")
    if point_count is None:
        raise ValueError(f"PLY vertex count is missing: {path}")
    if non_vertex_elements:
        raise ValueError(f"non-vertex PLY elements are unsupported: {non_vertex_elements}")
    if not properties:
        raise ValueError(f"PLY vertex properties are missing: {path}")

    dtype = np.dtype(properties, align=False)
    required = {"x", "y", "z", "intensity"}
    if not required.issubset(dtype.names or ()):
        raise ValueError(f"PLY is missing required properties {sorted(required)}: {path}")
    expected_size = point_offset + point_count * dtype.itemsize
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"unexpected PLY size for vertex-only data: expected {expected_size}, "
            f"got {path.stat().st_size}: {path}"
        )
    return PlyInfo(
        path=path,
        header_lines=tuple(header_lines),
        point_offset=point_offset,
        point_count=point_count,
        point_length=dtype.itemsize,
        dtype=dtype,
    )


def map_points(info: PlyInfo) -> np.memmap:
    return np.memmap(
        info.path,
        dtype=info.dtype,
        mode="r",
        offset=info.point_offset,
        shape=(info.point_count,),
    )


def world_positions(points: np.ndarray, _info: PlyInfo) -> np.ndarray:
    return np.column_stack((points["x"], points["y"], points["z"])).astype(np.float32, copy=False)


def normals(points: np.ndarray) -> np.ndarray | None:
    fields = set(points.dtype.names or ())
    if not {"normal_x", "normal_y", "normal_z"}.issubset(fields):
        return None
    result = np.column_stack((points["normal_x"], points["normal_y"], points["normal_z"])).astype(
        np.float32, copy=False
    )
    lengths = np.linalg.norm(result, axis=1)
    valid = np.isfinite(lengths) & (lengths > 1e-6)
    result[valid] /= lengths[valid, None]
    result[~valid] = 0
    return result


def _colored_header(source: PlyInfo, point_count: int) -> bytes:
    newline = b"\r\n" if source.header_lines[0].endswith(b"\r\n") else b"\n"
    result: list[bytes] = []
    for raw in source.header_lines:
        text = raw.decode("ascii").strip()
        if text.startswith("element vertex "):
            result.append(f"element vertex {point_count}".encode("ascii") + newline)
        elif text == "end_header":
            result.extend(
                (
                    b"comment RGB appended by native SHARE colorizer" + newline,
                    b"property uchar red" + newline,
                    b"property uchar green" + newline,
                    b"property uchar blue" + newline,
                    raw,
                )
            )
        else:
            result.append(raw)
    return b"".join(result)


def write_colored_ply(
    source: PlyInfo,
    output: Path,
    colors: np.ndarray,
    colored: np.ndarray,
    *,
    drop_uncolored: bool,
    fallback_rgb: tuple[int, int, int] = (255, 255, 255),
    block_size: int = 500_000,
) -> int:
    """Copy source records byte-for-byte, append RGB, and optionally omit points."""
    if colors.shape != (source.point_count, 3) or colors.dtype != np.uint8:
        raise ValueError("colors must be an N x 3 uint8 array")
    if colored.shape != (source.point_count,) or colored.dtype != np.bool_:
        raise ValueError("colored must be an N-element bool array")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output PLY: {output}")
    if {"red", "green", "blue"}.intersection(source.dtype.names or ()):
        raise ValueError("source PLY already contains RGB properties")

    keep = colored if drop_uncolored else np.ones(source.point_count, dtype=np.bool_)
    point_count = int(np.count_nonzero(keep))
    if point_count == 0:
        raise RuntimeError("no points survived colorization")

    raw_dtype = np.dtype((np.void, source.point_length))
    raw_source = np.memmap(
        source.path,
        dtype=raw_dtype,
        mode="r",
        offset=source.point_offset,
        shape=(source.point_count,),
    )
    fallback = np.asarray(fallback_rgb, dtype=np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(_colored_header(source, point_count))
        for start in range(0, source.point_count, block_size):
            stop = min(start + block_size, source.point_count)
            selected = keep[start:stop]
            source_bytes = np.frombuffer(raw_source[start:stop].tobytes(), dtype=np.uint8).reshape(
                stop - start, source.point_length
            )
            block_colors = colors[start:stop].copy()
            block_colors[~colored[start:stop]] = fallback
            records = np.empty((stop - start, source.point_length + 3), dtype=np.uint8)
            records[:, : source.point_length] = source_bytes
            records[:, source.point_length :] = block_colors
            stream.write(records[selected].tobytes())
        stream.flush()
        os.fsync(stream.fileno())
    return point_count
