"""Robust exposure fit and independent consensus-blend reference."""

import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, spsolve


def normal_matrix(edges, w, nodes):
    """A.T diag(w) A for difference equations x[i] - x[j], assembled directly from the edge list."""
    i, j = edges[:, 0], edges[:, 1]
    rows = np.concatenate([i, j, i, j])
    cols = np.concatenate([i, j, j, i])
    data = np.concatenate([w, w, -w, -w])
    return sparse.coo_matrix((data, (rows, cols)), shape=(nodes, nodes)).tocsr()


def solve_spd(matrix, rhs, start):
    """Solve a symmetric positive definite sparse system.

    Conjugate gradients with a Jacobi preconditioner, warm-started; a direct
    factorisation fills in badly once tens of thousands of cells from many
    photos are coupled, while each CG step is one cheap product.
    """
    diagonal = matrix.diagonal()
    inverse = np.where(diagonal > 0, 1.0 / np.maximum(diagonal, 1e-12), 1.0)
    preconditioner = LinearOperator(matrix.shape, matvec=lambda v: inverse * v, dtype=np.float64)
    solution, info = cg(matrix, rhs, x0=start, M=preconditioner, rtol=1e-9, maxiter=5000)
    if info != 0:
        # Fall back to the exact solve rather than return a partial answer.
        solution = spsolve(matrix, rhs)
    return solution


def solve_channel(args):
    """Iteratively reweighted least squares for one colour channel. Returns (correction, median residual)."""
    edges, y, base, reg, prior, local, nodes = args
    i, j = edges[:, 0], edges[:, 1]
    w = base.copy()
    correction = prior.copy()
    residual = np.zeros(0)
    for _iteration in range(4):
        wy = w * y
        # bincount returns integers when there are no equations; keep float64 either way.
        rhs = (np.bincount(i, wy, nodes) - np.bincount(j, wy, nodes)).astype(np.float64)
        if local:
            rhs += 8 * prior
        correction = solve_spd(normal_matrix(edges, w, nodes) + reg, rhs, correction)
        residual = correction[i] - correction[j] - y
        w = base * np.minimum(1, 8 / np.maximum(abs(residual), 1e-05))
    return correction, (float(np.median(abs(residual))) if len(residual) else None)


class _Serial:
    def map(self, fn, items):
        return [fn(x) for x in items]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def fork_pool(workers):
    """Forked worker pool so the large edge arrays are shared copy-on-write; serial where fork is unavailable."""
    import multiprocessing

    try:
        return multiprocessing.get_context("fork").Pool(workers)
    except (ValueError, OSError):
        return _Serial()


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

    def sample(self, step=16, chunk_points=1 << 20):
        """Every step-th candidate record, read sequentially in chunks.

        A strided memmap read touches every page of the file and keeps it
        resident; streaming keeps memory near the sample size.
        """
        n = json.loads((self.output / "candidates/meta.json").read_text())["points"]
        record = self.k * 8
        parts = []
        with (self.output / "candidates/observations.bin").open("rb") as stream:
            for start in range(0, n, chunk_points):
                count = min(chunk_points, n - start)
                block = np.fromfile(stream, dtype="float32", count=count * record).reshape(
                    count, self.k, 8
                )
                first = (-start) % step
                parts.append(block[first::step].copy())
        return np.concatenate(parts) if parts else np.zeros((0, self.k, 8), dtype="float32")

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
        sample = self.sample()
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
        prior = np.zeros((nodes, 3))
        if mode == "local":
            global_field = np.load(self.output / "global/field.npy")
            prior = global_field.reshape(nodes, 3)
            # Smoothness between neighbouring cells of the same image.
            cell = np.arange(nodes).reshape(self.image_count, self.gh, self.gw)
            right = np.column_stack([cell[:, :, :-1].ravel(), cell[:, :, 1:].ravel()])
            down = np.column_stack([cell[:, :-1, :].ravel(), cell[:, 1:, :].ravel()])
            pairs = np.concatenate([right, down])
            reg = sparse.eye(nodes, format="csr") * 8 + normal_matrix(
                pairs, np.full(len(pairs), 80.0), nodes
            )
        else:
            reg = sparse.eye(nodes, format="csr") * 0.15
        # Each channel is an independent robust fit; run them side by side.
        with fork_pool(3) as pool:
            solved = pool.map(
                solve_channel,
                [
                    (edges, y[:, channel], base, reg, prior[:, channel], mode == "local", nodes)
                    for channel in range(3)
                ],
            )
        correction = np.column_stack([s[0] for s in solved])
        errors = [s[1] for s in solved]
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
            "channel_median_residual": errors,
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
