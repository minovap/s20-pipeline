"""Robust exposure fit and independent consensus-blend reference."""

import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


class Exposure:
    k = 4
    gw = 8
    gh = 6

    def __init__(self, output, image_count):
        self.output = output
        self.image_count = image_count

    def candidates(self):
        n = json.loads((self.output / "candidates/meta.json").read_text())["points"]
        return np.memmap(
            self.output / "candidates/observations.bin",
            dtype="float32",
            mode="r",
            shape=(n, self.k, 8),
        )

    def interpolate(self, c, field):
        x = c[..., 4]
        y = c[..., 5]
        ix = x.astype("int32")
        iy = y.astype("int32")
        j = c[..., 6].astype("int32")
        jx = np.minimum(ix + 1, self.gw - 1)
        jy = np.minimum(iy + 1, self.gh - 1)
        a = (x - ix)[..., None]
        b = (y - iy)[..., None]
        return (
            (1 - a) * (1 - b) * field[j, iy, ix]
            + a * (1 - b) * field[j, iy, jx]
            + (1 - a) * b * field[j, jy, ix]
            + a * b * field[j, jy, jx]
        )

    def solve(self, mode):
        dest = self.output / mode
        dest.mkdir(parents=True, exist_ok=False)
        allc = self.candidates()
        sample = np.asarray(allc[::16])
        a = sample[:, 0]
        indices = np.arange(len(sample))
        train = indices % 5 != 0
        equations = []
        observations = []
        weights = []
        validation = []
        nodes = self.image_count if mode == "global" else self.image_count * self.gw * self.gh

        def node(c):
            image = c[:, 6].astype(int)
            if mode == "global":
                return image
            return (
                image * self.gw * self.gh
                + np.rint(c[:, 5]).astype(int) * self.gw
                + np.rint(c[:, 4]).astype(int)
            )

        for k in range(1, self.k):
            b = sample[:, k]
            delta = a[:, :3] - b[:, :3]
            good = (
                (a[:, 7] > 0)
                & (b[:, 7] > 0.1 * a[:, 7])
                & (a[:, 3] < 22)
                & (b[:, 3] < 22)
                & (a[:, :3].min(1) > 12)
                & (b[:, :3].min(1) > 12)
                & (a[:, :3].max(1) < 243)
                & (b[:, :3].max(1) < 243)
                & (abs(delta).max(1) < 40)
            )
            i = node(b)
            j = node(a)
            keep = good & train
            equations.append(np.column_stack([i[keep], j[keep]]))
            observations.append(delta[keep])
            weights.append(np.sqrt(b[keep, 7] / a[keep, 7]))
            validation.append((a[good & ~train], b[good & ~train]))
        edges = np.concatenate(equations)
        y = np.concatenate(observations)
        base = np.concatenate(weights)
        count = len(y)
        ri = np.repeat(np.arange(count), 2)
        A = sparse.coo_matrix(
            (np.tile([1.0, -1.0], count), (ri, edges.ravel())), shape=(count, nodes)
        ).tocsr()
        reg = sparse.eye(nodes, format="csr") * 0.15
        prior = np.zeros((nodes, 3))
        if mode == "local":
            global_field = np.load(self.output / "global/field.npy")
            prior = global_field.reshape(nodes, 3)
            pairs = []
            for im in range(self.image_count):
                for yy in range(self.gh):
                    for xx in range(self.gw):
                        n = im * self.gw * self.gh + yy * self.gw + xx
                        if xx + 1 < self.gw:
                            pairs.append((n, n + 1))
                        if yy + 1 < self.gh:
                            pairs.append((n, n + self.gw))
            pairs = np.array(pairs)
            D = sparse.coo_matrix(
                (
                    np.tile([1.0, -1.0], len(pairs)),
                    (np.repeat(np.arange(len(pairs)), 2), pairs.ravel()),
                ),
                shape=(len(pairs), nodes),
            ).tocsr()
            reg = sparse.eye(nodes, format="csr") * 8 + D.T @ D * 80
        correction = prior.copy()
        errors = []
        for channel in range(3):
            w = base.copy()
            for iteration in range(4):
                W = A.multiply(w[:, None])
                rhs = A.T @ (w * y[:, channel])
                if mode == "local":
                    rhs += 8 * prior[:, channel]
                correction[:, channel] = spsolve(A.T @ W + reg, rhs)
                residual = A @ correction[:, channel] - y[:, channel]
                w = base * np.minimum(1, 8 / np.maximum(abs(residual), 1e-05))
            errors.append(float(np.median(abs(residual))) if len(residual) else None)
        correction = np.clip(correction, -32, 32).astype("float32")
        if mode == "global":
            field = np.broadcast_to(
                correction[:, None, None, :], (self.image_count, self.gh, self.gw, 3)
            ).copy()
        else:
            field = correction.reshape(self.image_count, self.gh, self.gw, 3)
        np.save(dest / "field.npy", field)
        field.tofile(dest / "field.bin")
        aa = np.concatenate([p[0] for p in validation])
        bb = np.concatenate([p[1] for p in validation])
        before = abs(aa[:, :3] - bb[:, :3])
        after = abs(
            aa[:, :3] + self.interpolate(aa, field) - bb[:, :3] - self.interpolate(bb, field)
        )
        result = {
            "mode": mode,
            "training_pairs": count,
            "heldout_pairs": len(aa),
            "heldout_before_channel_median": float(np.median(before)) if len(aa) else None,
            "heldout_after_channel_median": float(np.median(after)) if len(aa) else None,
            "heldout_before_channel_p95": float(np.percentile(before, 95)) if len(aa) else None,
            "heldout_after_channel_p95": float(np.percentile(after, 95)) if len(aa) else None,
            "offset_min": float(field.min()),
            "offset_max": float(field.max()),
            "method": "Robust additive RGB offsets from same-point overlapping views; low-gradient/non-saturated guards; zero prior for global, global prior and neighbor smoothness for local; ±32 channel clamp. Local nearest-cell fit with bilinear application. Heldout split by point, not image pair. No Studio colors used.",
        }
        (dest / "solve.json").write_text(json.dumps(result, indent=2))
        print(result, flush=True)

    def reference(self, c, field, robust=False):
        x = c[..., 4]
        y = c[..., 5]
        ix = x.astype("int32")
        iy = y.astype("int32")
        im = c[..., 6].astype("int32")
        wx = np.floor((x - ix) * 256 + 0.5).astype("int32")[..., None]
        wy = np.floor((y - iy) * 256 + 0.5).astype("int32")[..., None]
        q = np.floor(field * 256 + 0.5).astype("int32")
        jx = np.minimum(ix + 1, 7)
        jy = np.minimum(iy + 1, 5)
        offset = (
            (256 - wx) * (256 - wy) * q[im, iy, ix]
            + wx * (256 - wy) * q[im, iy, jx]
            + (256 - wx) * wy * q[im, jy, ix]
            + wx * wy * q[im, jy, jx]
            + 32768
        ) // 65536
        rgb = np.clip(np.floor(c[..., :3] * 256 + 0.5) + offset, 0, 65280) / 65280
        if robust:
            quant = np.floor(rgb * 65280 + 0.5).astype("int32")
            valid = (c[:, :, 7] > 0) & (c[:, :, 7] >= 0.1 * c[:, 0:1, 7]) & (c[:, :, 3] < 22)
            pair = np.minimum(abs(quant[:, :, None, :] - quant[:, None, :, :]).max(3), 12288)
            cost = (pair * valid[:, None, :]).sum(2)
            cost[~valid] = 2147483647
            chosen = np.argmin(cost, axis=1)
            chosen[c[:, 0, 3] >= 22] = 0
            anchor = rgb[np.arange(len(c)), chosen]
            delta = abs(quant - quant[np.arange(len(c)), chosen][:, None, :]).max(2) / 256
            good = valid & (c[np.arange(len(c)), chosen, 3:4] < 22) & (delta < 48)
            good[np.arange(len(c)), chosen] = c[np.arange(len(c)), chosen, 7] > 0
            taper = np.maximum(0, 1 - delta * delta / (48 * 48))
            w = (
                np.sqrt(np.maximum(c[:, :, 7], 0) / np.maximum(c[:, 0:1, 7], 1e-20))
                * good
                * taper
                * taper
                / (1 + c[:, :, 3] ** 2 / (22 * 22))
            )
            linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            mean = (linear * w[:, :, None]).sum(1) / np.maximum(w.sum(1)[:, None], 1e-20)
            srgb = (
                np.where(
                    mean <= 0.0031308,
                    mean * 12.92,
                    1.055 * np.maximum(mean, 0) ** (1 / 2.4) - 0.055,
                )
                * 255
            )
            return np.column_stack([srgb, (good.sum(1) > 1).astype(float)]).astype("float32")
        anchor = rgb[:, 0, :]
        good = (
            (c[:, :, 7] > 0)
            & (c[:, :, 7] >= 0.1 * c[:, 0:1, 7])
            & (c[:, :, 3] < 22)
            & (c[:, 0:1, 3] < 22)
            & (np.floor(abs(rgb - anchor[:, None, :]).max(2) * 65280 + 0.5) <= 6272)
        )
        good[:, 0] = c[:, 0, 7] > 0
        w = np.sqrt(np.maximum(c[:, :, 7], 0) / np.maximum(c[:, 0:1, 7], 1e-20)) * good
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        mean = (linear * w[:, :, None]).sum(1) / np.maximum(w.sum(1)[:, None], 1e-20)
        srgb = (
            np.where(
                mean <= 0.0031308, mean * 12.92, 1.055 * np.maximum(mean, 0) ** (1 / 2.4) - 0.055
            )
            * 255
        )
        return np.column_stack([srgb, (good.sum(1) > 1).astype(float)]).astype("float32")
