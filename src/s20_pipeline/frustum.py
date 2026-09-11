"""Conservative voxel bounds for the collector's polynomial fisheye cameras."""

import numpy as np


def _product(lo, hi, other_lo, other_hi):
    values = np.asarray([lo * other_lo, lo * other_hi, hi * other_lo, hi * other_hi])
    return values.min(axis=0), values.max(axis=0)


class PhotoPointIndex:
    """A sorted 2 m voxel index; candidate IDs remain in original point order."""

    def __init__(self, xyz, voxel_size=2.0, chunk=262144):
        self.size = voxel_size
        self.count = len(xyz)
        self.last_selection = None
        lower = np.floor(xyz.min(axis=0).astype(np.float64) / voxel_size)
        upper = np.floor(xyz.max(axis=0).astype(np.float64) / voxel_size)
        shape = tuple(int(value) + 1 for value in upper - lower)
        # Extremely large coordinates cannot be encoded safely. Retain everything.
        if np.prod(shape, dtype=object) > np.iinfo(np.int64).max:
            self.order = None
            return
        keys = np.empty(len(xyz), dtype=np.int64)
        for start in range(0, len(xyz), chunk):
            cells = (
                np.floor(xyz[start : start + chunk].astype(np.float64) / voxel_size) - lower
            ).astype(np.int64)
            keys[start : start + chunk] = (cells[:, 0] * shape[1] + cells[:, 1]) * shape[2] + cells[
                :, 2
            ]
        self.order = np.argsort(keys, kind="stable")
        if self.count <= np.iinfo(np.uint32).max:
            self.order = self.order.astype(np.uint32)
        keys = keys[self.order]
        self.starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
        self.counts = np.diff(np.r_[self.starts, len(xyz)])
        if self.count <= np.iinfo(np.uint32).max:
            self.starts = self.starts.astype(np.uint32)
        occupied = keys[self.starts]
        cells = np.column_stack(np.unravel_index(occupied, shape))
        self.centers = (cells + lower + 0.5) * voxel_size

    def visible_voxels(self, frame):
        """Bound all eight corners, including a one-voxel halo on every side.

        Corner *projections* alone are unsafe: a fisheye image can intersect a
        box without containing any corner. Instead bound the camera-space box,
        angle, polynomial radius and pixel coordinates with interval arithmetic.
        Nonmonotonic distortion and cameras inside a voxel remain conservative.
        There is no far clipping plane: collect() only imposes d > 0.1.
        """
        rotation = frame.camera_to_world.astype(np.float32).astype(np.float64)
        camera = (self.centers - frame.center.astype(np.float32)) @ rotation
        extent = 1.5 * self.size * np.abs(rotation).sum(axis=0)
        # Cover float32 subtraction/matmul rounding, also far from the origin.
        error = (
            32
            * np.finfo(np.float32).eps
            * (
                (np.abs(self.centers) + np.abs(frame.center.astype(np.float32))) @ np.abs(rotation)
                + extent
                + 1
            )
        )
        lo, hi = camera - extent - error, camera + extent + error
        radial_lo = np.linalg.norm(np.maximum(np.maximum(lo[:, :2], -hi[:, :2]), 0), axis=1)
        radial_hi = np.linalg.norm(np.maximum(abs(lo[:, :2]), abs(hi[:, :2])), axis=1)
        angles = np.asarray(
            [
                np.arctan2(radial_lo, lo[:, 2]),
                np.arctan2(radial_lo, hi[:, 2]),
                np.arctan2(radial_hi, lo[:, 2]),
                np.arctan2(radial_hi, hi[:, 2]),
            ]
        )
        theta_lo, theta_hi = angles.min(axis=0), angles.max(axis=0)
        limit = np.deg2rad(frame.calibration.max_incident_angle_deg)
        keep = theta_lo <= limit + 1e-5
        theta_hi = np.maximum(theta_lo, np.minimum(theta_hi, limit + 1e-5))
        # Match theta + k2*theta**2 + ... + k7*theta**7 without assuming
        # the polynomial is positive or monotonic for a supplied calibration.
        # Horner intervals are tighter than bounding each power separately,
        # especially where the calibration's large signed terms cancel.
        coefficients = (0.0, 1.0, *frame.calibration.coefficients)
        dist_lo = dist_hi = np.full_like(theta_lo, coefficients[-1])
        for coefficient in coefficients[-2::-1]:
            dist_lo, dist_hi = _product(dist_lo, dist_hi, theta_lo, theta_hi)
            dist_lo += coefficient
            dist_hi += coefficient
        magnitude = theta_hi.copy()
        for power, coefficient in enumerate(frame.calibration.coefficients, 2):
            magnitude += abs(coefficient) * theta_hi**power
        error = 64 * np.finfo(np.float32).eps * (1 + magnitude)
        dist_lo -= error
        dist_hi += error
        normalized = []
        for axis in (0, 1):
            # The axis singularity uses scale=1 in project(); retain its box.
            with np.errstate(divide="ignore", invalid="ignore"):
                values = np.asarray(
                    [
                        lo[:, axis] / radial_lo,
                        lo[:, axis] / radial_hi,
                        hi[:, axis] / radial_lo,
                        hi[:, axis] / radial_hi,
                    ]
                )
            direction_lo = np.where(radial_lo > 1e-10, np.clip(values.min(axis=0), -1, 1), -1)
            direction_hi = np.where(radial_lo > 1e-10, np.clip(values.max(axis=0), -1, 1), 1)
            normalized.append(_product(direction_lo, direction_hi, dist_lo, dist_hi))
        cal = frame.calibration
        xlo, xhi = normalized[0]
        ylo, yhi = normalized[1]
        ulo, uhi = _product(xlo, xhi, cal.a11, cal.a11)
        skew_lo, skew_hi = _product(ylo, yhi, cal.a12, cal.a12)
        vlo, vhi = _product(ylo, yhi, cal.a22, cal.a22)
        # A pixel of padding covers the final float32 affine operations.
        keep &= (ulo + skew_lo + cal.u0 < cal.width) & (uhi + skew_hi + cal.u0 >= -1)
        keep &= (vlo + cal.v0 < cal.height) & (vhi + cal.v0 >= -1)
        # Never reject an interval whose arithmetic overflowed.
        finite = np.isfinite(np.column_stack([ulo, uhi, vlo, vhi, skew_lo, skew_hi])).all(axis=1)
        return keep | ~finite

    def point_ids(self, frame):
        """Return ordered IDs, or None for a dense view's faster contiguous path."""
        if self.order is None:
            self.last_selection = {
                "path": "unindexed",
                "occupied_voxels": 0,
                "retained_voxels": 0,
                "selected_points": self.count,
            }
            return None
        keep = self.visible_voxels(frame)
        selected_count = int(self.counts[keep].sum())
        # A compact indoor cloud often fits almost entirely in the halo. Avoid
        # gathering/copying it when culling would save fewer than 10% of points.
        if selected_count >= 0.9 * self.count:
            self.last_selection = {
                "path": "dense",
                "occupied_voxels": int(len(self.counts)),
                "retained_voxels": int(np.count_nonzero(keep)),
                "selected_points": self.count,
            }
            return None
        if selected_count <= self.count // 4:
            # Sparse views need only their selected IDs, not a full-cloud mask.
            # Broader views retain the linear mask path to avoid a large sort.
            selected = np.empty(selected_count, dtype=self.order.dtype)
            cursor = 0
            for leaf in np.flatnonzero(keep):
                start = int(self.starts[leaf])
                count = int(self.counts[leaf])
                selected[cursor : cursor + count] = self.order[start : start + count]
                cursor += count
            # Every ID is unique; ascending order exactly matches flatnonzero.
            selected.sort()
            self.last_selection = {
                "path": "sparse",
                "occupied_voxels": int(len(self.counts)),
                "retained_voxels": int(np.count_nonzero(keep)),
                "selected_points": selected_count,
            }
            return selected
        selected = np.empty(self.count, dtype=bool)
        selected[self.order] = np.repeat(keep, self.counts)
        point_ids = np.flatnonzero(selected)
        self.last_selection = {
            "path": "mask",
            "occupied_voxels": int(len(self.counts)),
            "retained_voxels": int(np.count_nonzero(keep)),
            "selected_points": selected_count,
        }
        return point_ids
