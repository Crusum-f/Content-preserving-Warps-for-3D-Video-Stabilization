from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr

from .geometry import apply_homography


@dataclass
class WarpResult:
    image: np.ndarray
    mask: np.ndarray
    output_vertices: np.ndarray
    source_vertices: np.ndarray
    num_constraints: int


def make_grid(width: int, height: int, cols: int, rows: int) -> np.ndarray:
    xs = np.linspace(0.0, width - 1.0, cols + 1)
    ys = np.linspace(0.0, height - 1.0, rows + 1)
    xv, yv = np.meshgrid(xs, ys)
    return np.stack([xv, yv], axis=-1).reshape(-1, 2)


def salience_weights(image: np.ndarray, cols: int, rows: int, eps: float = 0.5) -> np.ndarray:
    img = image.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    weights = np.empty((rows, cols), dtype=np.float64)
    for j in range(rows):
        y0 = int(round(j * h / rows))
        y1 = int(round((j + 1) * h / rows))
        for i in range(cols):
            x0 = int(round(i * w / cols))
            x1 = int(round((i + 1) * w / cols))
            cell = img[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
            weights[j, i] = float(np.linalg.norm(cell.reshape(-1, cell.shape[-1]).var(axis=0)) + eps)
    return weights


def salience_weights_for_grid(image: np.ndarray, grid: np.ndarray, cols: int, rows: int, eps: float = 0.5) -> np.ndarray:
    h, w = image.shape[:2]
    base = make_grid(w, h, cols, rows)
    if np.allclose(grid, base, atol=1e-9):
        return salience_weights(image, cols, rows, eps=eps)

    img = image.astype(np.float32) / 255.0
    weights = np.empty((rows, cols), dtype=np.float64)
    for j in range(rows):
        for i in range(cols):
            v00 = j * (cols + 1) + i
            v10 = v00 + 1
            v01 = v00 + (cols + 1)
            v11 = v01 + 1
            quad = grid[[v00, v10, v11, v01]].astype(np.float32)
            x, y, bw, bh = cv2.boundingRect(quad)
            x0 = max(x, 0)
            y0 = max(y, 0)
            x1 = min(x + bw, w)
            y1 = min(y + bh, h)
            if x0 >= x1 or y0 >= y1:
                weights[j, i] = eps
                continue
            roi_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            shifted = quad - np.array([x0, y0], dtype=np.float32)
            cv2.fillConvexPoly(roi_mask, np.round(shifted).astype(np.int32), 255)
            pixels = img[y0:y1, x0:x1][roi_mask > 0]
            if len(pixels) == 0:
                weights[j, i] = eps
            else:
                weights[j, i] = float(np.linalg.norm(pixels.var(axis=0)) + eps)
    return weights


def bilinear_cells_and_weights(points: np.ndarray, width: int, height: int, cols: int, rows: int):
    cell_w = (width - 1.0) / cols
    cell_h = (height - 1.0) / rows
    gx = points[:, 0] / cell_w
    gy = points[:, 1] / cell_h
    ix = np.floor(gx).astype(np.int64)
    iy = np.floor(gy).astype(np.int64)
    valid = (ix >= 0) & (iy >= 0) & (ix < cols) & (iy < rows)
    ix = np.clip(ix, 0, cols - 1)
    iy = np.clip(iy, 0, rows - 1)
    tx = np.clip(gx - ix, 0.0, 1.0)
    ty = np.clip(gy - iy, 0.0, 1.0)
    weights = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty], axis=1)
    v00 = iy * (cols + 1) + ix
    v10 = v00 + 1
    v01 = v00 + (cols + 1)
    v11 = v01 + 1
    vertices = np.stack([v00, v10, v01, v11], axis=1)
    return vertices[valid], weights[valid], valid


def bilinear_weights_in_warped_grid(
    original_points: np.ndarray,
    warped_points: np.ndarray,
    source_grid: np.ndarray,
    width: int,
    height: int,
    cols: int,
    rows: int,
):
    cell_w = (width - 1.0) / cols
    cell_h = (height - 1.0) / rows
    gx = original_points[:, 0] / cell_w
    gy = original_points[:, 1] / cell_h
    ix = np.floor(gx).astype(np.int64)
    iy = np.floor(gy).astype(np.int64)
    valid = (ix >= 0) & (iy >= 0) & (ix < cols) & (iy < rows) & np.isfinite(warped_points).all(axis=1)
    ix = np.clip(ix, 0, cols - 1)
    iy = np.clip(iy, 0, rows - 1)

    v00 = iy * (cols + 1) + ix
    v10 = v00 + 1
    v01 = v00 + (cols + 1)
    v11 = v01 + 1
    vertices = np.stack([v00, v10, v01, v11], axis=1)

    weights = np.zeros((len(original_points), 4), dtype=np.float64)
    quads = source_grid[vertices]
    for idx in np.flatnonzero(valid):
        u, v = _invert_bilinear_quad(quads[idx], warped_points[idx])
        weights[idx] = [(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v]
    return vertices[valid], weights[valid], valid


def solve_content_preserving_warp(
    image: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    data_weights: np.ndarray,
    cols: int = 64,
    rows: int = 36,
    alpha: float = 20.0,
    anchor_weight: float = 0.0,
    homography: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    h, w = image.shape[:2]
    base_grid = make_grid(w, h, cols, rows)
    if homography is None:
        source_grid = base_grid.copy()
        source_for_weights = source_points
    else:
        source_grid = apply_homography(base_grid, homography)
        source_for_weights = apply_homography(source_points, homography)

    nverts = source_grid.shape[0]
    rhs: list[float] = []
    row_idx: list[int] = []
    col_idx: list[int] = []
    values: list[float] = []
    row = 0

    def add_coeff(r: int, vid: int, axis: int, coeff: float) -> None:
        row_idx.append(r)
        col_idx.append(2 * int(vid) + axis)
        values.append(float(coeff))

    if homography is None:
        v_ids, wts, valid = bilinear_cells_and_weights(source_points, w, h, cols, rows)
    else:
        v_ids, wts, valid = bilinear_weights_in_warped_grid(source_points, source_for_weights, source_grid, w, h, cols, rows)
    targets = target_points[valid]
    dweights = np.sqrt(np.maximum(data_weights[valid], 0.0))
    data_row_count = 0
    for verts, bw, target, dw in zip(v_ids, wts, targets, dweights):
        if dw <= 0:
            continue
        for axis in (0, 1):
            for vid, b in zip(verts, bw):
                add_coeff(row, vid, axis, dw * b)
            rhs.append(float(dw * target[axis]))
            row += 1
            data_row_count += 1

    if data_row_count == 0:
        return source_grid.copy(), source_grid, 0

    sal = salience_weights_for_grid(image, source_grid, cols, rows)
    for a, b, c, cell_ij in _vertex_centered_triangle_constraints(cols, rows):
        pa, pb, pc = source_grid[[a, b, c]]
        cell_weight = sal[cell_ij[1], cell_ij[0]]
        smooth_weight = float(np.sqrt(alpha * cell_weight))
        u, v = _local_similarity_coordinates(pa, pb, pc)
        # x_a - (x_b + u(x_c-x_b) + v(y_c-y_b)) = 0
        add_coeff(row, a, 0, smooth_weight)
        add_coeff(row, b, 0, smooth_weight * (u - 1.0))
        add_coeff(row, c, 0, smooth_weight * (-u))
        add_coeff(row, b, 1, smooth_weight * v)
        add_coeff(row, c, 1, smooth_weight * (-v))
        rhs.append(0.0)
        row += 1

        # y_a - (y_b + u(y_c-y_b) + v(x_b-x_c)) = 0
        add_coeff(row, a, 1, smooth_weight)
        add_coeff(row, b, 1, smooth_weight * (u - 1.0))
        add_coeff(row, c, 1, smooth_weight * (-u))
        add_coeff(row, b, 0, smooth_weight * (-v))
        add_coeff(row, c, 0, smooth_weight * v)
        rhs.append(0.0)
        row += 1

    if anchor_weight > 0:
        aw = float(np.sqrt(anchor_weight))
        for vid, pt in enumerate(source_grid):
            add_coeff(row, vid, 0, aw)
            rhs.append(float(aw * pt[0]))
            row += 1
            add_coeff(row, vid, 1, aw)
            rhs.append(float(aw * pt[1]))
            row += 1

    A = sparse.coo_matrix((values, (row_idx, col_idx)), shape=(row, 2 * nverts)).tocsr()
    b = np.asarray(rhs, dtype=np.float64)
    x = lsmr(A, b, atol=1e-7, btol=1e-7, maxiter=400)[0]
    return x.reshape(nverts, 2), source_grid, int(valid.sum())


def render_warped_mesh(image: np.ndarray, source_vertices: np.ndarray, output_vertices: np.ndarray, cols: int, rows: int):
    h, w = image.shape[:2]
    out = np.zeros_like(image)
    mask = np.zeros((h, w), dtype=np.uint8)
    for tri, _cell in _grid_triangles(cols, rows):
        src = source_vertices[np.asarray(tri)].astype(np.float32)
        dst = output_vertices[np.asarray(tri)].astype(np.float32)

        x, y, bw, bh = cv2.boundingRect(dst)
        x0 = max(x, 0)
        y0 = max(y, 0)
        x1 = min(x + bw, w)
        y1 = min(y + bh, h)
        if x0 >= x1 or y0 >= y1:
            continue

        inv_M = cv2.getAffineTransform(dst, src)
        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        map_x = (inv_M[0, 0] * grid_x + inv_M[0, 1] * grid_y + inv_M[0, 2]).astype(np.float32)
        map_y = (inv_M[1, 0] * grid_x + inv_M[1, 1] * grid_y + inv_M[1, 2]).astype(np.float32)
        patch = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        roi_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        shifted_dst = dst - np.array([x0, y0], dtype=np.float32)
        cv2.fillConvexPoly(roi_mask, np.round(shifted_dst).astype(np.int32), 255)
        valid_sample = (map_x >= 0.0) & (map_x <= w - 1.0) & (map_y >= 0.0) & (map_y <= h - 1.0)
        valid_roi = (roi_mask > 0) & valid_sample
        roi = out[y0:y1, x0:x1]
        roi[valid_roi] = patch[valid_roi]
        mask[y0:y1, x0:x1][valid_roi] = 255
    return out, mask


def warp_frame(
    image: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    data_weights: np.ndarray,
    cols: int = 64,
    rows: int = 36,
    alpha: float = 20.0,
    anchor_weight: float = 0.0,
    homography: np.ndarray | None = None,
    input_mask: np.ndarray | None = None,
) -> WarpResult:
    if homography is None:
        working = image
        working_mask = input_mask
    else:
        h, w = image.shape[:2]
        working = cv2.warpPerspective(image, homography, (w, h), flags=cv2.INTER_LINEAR)
        working_mask = None
        if input_mask is not None:
            working_mask = cv2.warpPerspective(input_mask, homography, (w, h), flags=cv2.INTER_NEAREST)
    output_vertices, source_vertices, n = solve_content_preserving_warp(
        working,
        source_points,
        target_points,
        data_weights,
        cols=cols,
        rows=rows,
        alpha=alpha,
        anchor_weight=anchor_weight,
        homography=homography,
    )
    out, mask = render_warped_mesh(working, source_vertices, output_vertices, cols, rows)
    if working_mask is not None:
        warped_valid, sample_mask = render_warped_mesh(working_mask, source_vertices, output_vertices, cols, rows)
        mask = np.where((mask > 0) & (sample_mask > 0) & (warped_valid > 127), 255, 0).astype(np.uint8)
    return WarpResult(out, mask, output_vertices, source_vertices, n)


def estimate_homography(
    source_points: np.ndarray,
    target_points: np.ndarray,
    weights: np.ndarray,
    min_points: int = 4,
    ransac_max_error: float = 0.0,
):
    valid = np.isfinite(source_points).all(axis=1) & np.isfinite(target_points).all(axis=1) & (weights > 0)
    if int(valid.sum()) < min_points:
        return None
    src = source_points[valid].astype(np.float64)
    dst = target_points[valid].astype(np.float64)
    w = weights[valid].astype(np.float64)

    if ransac_max_error > 0:
        H, inliers = cv2.findHomography(
            src.astype(np.float32),
            dst.astype(np.float32),
            cv2.RANSAC,
            ransacReprojThreshold=float(ransac_max_error),
        )
        if H is None or inliers is None:
            return None
        keep = inliers.ravel().astype(bool)
        if int(keep.sum()) < min_points:
            return None
        src = src[keep]
        dst = dst[keep]
        w = w[keep]

    return _weighted_homography_dlt(src, dst, w)


def _weighted_homography_dlt(source_points: np.ndarray, target_points: np.ndarray, weights: np.ndarray):
    src_n, T_src = _normalize_points(source_points)
    dst_n, T_dst = _normalize_points(target_points)
    rows = []
    for (x, y), (u, v), weight in zip(src_n, dst_n, np.sqrt(np.maximum(weights, 0.0))):
        if weight <= 0:
            continue
        rows.append(weight * np.array([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u], dtype=np.float64))
        rows.append(weight * np.array([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v], dtype=np.float64))
    if len(rows) < 8:
        return None
    A = np.stack(rows, axis=0)
    try:
        _u, _s, vh = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    Hn = vh[-1].reshape(3, 3)
    H = np.linalg.inv(T_dst) @ Hn @ T_src
    if abs(H[2, 2]) < 1e-12:
        return None
    return H / H[2, 2]


def _normalize_points(points: np.ndarray):
    mean = points.mean(axis=0)
    centered = points - mean
    rms = np.sqrt(np.maximum((centered * centered).sum(axis=1).mean(), 1e-12))
    scale = np.sqrt(2.0) / rms
    T = np.array([[scale, 0.0, -scale * mean[0]], [0.0, scale, -scale * mean[1]], [0.0, 0.0, 1.0]])
    pts_h = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    norm = pts_h @ T.T
    return norm[:, :2], T


def _invert_bilinear_quad(quad: np.ndarray, point: np.ndarray) -> tuple[float, float]:
    p00, p10, p01, p11 = quad
    u = 0.5
    v = 0.5
    for _ in range(8):
        q = (
            (1.0 - u) * (1.0 - v) * p00
            + u * (1.0 - v) * p10
            + (1.0 - u) * v * p01
            + u * v * p11
        )
        du = -(1.0 - v) * p00 + (1.0 - v) * p10 - v * p01 + v * p11
        dv = -(1.0 - u) * p00 - u * p10 + (1.0 - u) * p01 + u * p11
        J = np.stack([du, dv], axis=1)
        residual = q - point
        try:
            step = np.linalg.solve(J, residual)
        except np.linalg.LinAlgError:
            break
        u -= float(step[0])
        v -= float(step[1])
        if float(step @ step) < 1e-18:
            break
    return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))


def _grid_triangles(cols: int, rows: int):
    tris = []
    for j in range(rows):
        for i in range(cols):
            v00 = j * (cols + 1) + i
            v10 = v00 + 1
            v01 = v00 + (cols + 1)
            v11 = v01 + 1
            tris.append(((v00, v10, v11), (i, j)))
            tris.append(((v00, v11, v01), (i, j)))
    return tris


def _vertex_centered_triangle_constraints(cols: int, rows: int):
    constraints = []
    for j in range(rows):
        for i in range(cols):
            v00 = j * (cols + 1) + i
            v10 = v00 + 1
            v01 = v00 + (cols + 1)
            v11 = v01 + 1
            constraints.extend(
                [
                    (v00, v10, v11, (i, j)),
                    (v00, v11, v01, (i, j)),
                    (v10, v11, v01, (i, j)),
                    (v10, v01, v00, (i, j)),
                    (v11, v01, v00, (i, j)),
                    (v11, v00, v10, (i, j)),
                    (v01, v00, v10, (i, j)),
                    (v01, v10, v11, (i, j)),
                ]
            )
    return constraints


def _local_similarity_coordinates(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> tuple[float, float]:
    edge = pc - pb
    basis = np.array([[edge[0], edge[1]], [edge[1], -edge[0]]], dtype=np.float64)
    rhs = pa - pb
    det = float(np.linalg.det(basis))
    if abs(det) < 1e-12:
        return 0.0, 0.0
    u, v = np.linalg.solve(basis, rhs)
    return float(u), float(v)
