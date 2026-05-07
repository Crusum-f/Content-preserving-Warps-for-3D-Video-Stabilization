from __future__ import annotations

import numpy as np
import torch

from .colmap import Camera


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64)
    K = np.array(
        [
            [m[0, 0] - m[1, 1] - m[2, 2], 0.0, 0.0, 0.0],
            [m[1, 0] + m[0, 1], m[1, 1] - m[0, 0] - m[2, 2], 0.0, 0.0],
            [m[2, 0] + m[0, 2], m[2, 1] + m[1, 2], m[2, 2] - m[0, 0] - m[1, 1], 0.0],
            [m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0], m[0, 0] + m[1, 1] + m[2, 2]],
        ],
        dtype=np.float64,
    )
    K /= 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    if q[0] < 0:
        q *= -1
    return q


def camera_center_from_extrinsics(Rcw: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    return -Rcw.T @ np.asarray(tvec, dtype=np.float64)


def tvec_from_center(Rcw: np.ndarray, center: np.ndarray) -> np.ndarray:
    return -Rcw @ np.asarray(center, dtype=np.float64)


def project_points_numpy(
    xyz: np.ndarray,
    camera: Camera,
    Rcw: np.ndarray,
    tvec: np.ndarray,
    apply_distortion: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    pts = (Rcw @ xyz.T).T + tvec[None, :]
    z = pts[:, 2]
    valid = z > 1e-8
    x = pts[:, 0] / np.maximum(z, 1e-8)
    y = pts[:, 1] / np.maximum(z, 1e-8)
    if apply_distortion and camera.radial_k1 != 0.0:
        r2 = x * x + y * y
        factor = 1.0 + camera.radial_k1 * r2
        x = x * factor
        y = y * factor
    K = camera.K
    uv = np.stack([K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]], axis=1)
    valid &= np.isfinite(uv).all(axis=1)
    return uv, valid


def project_points_torch(
    xyz: torch.Tensor,
    K: torch.Tensor,
    Rcw: torch.Tensor,
    tvec: torch.Tensor,
    k1: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    pts = xyz @ Rcw.T + tvec[None, :]
    z = pts[:, 2]
    valid = z > 1e-8
    xy = pts[:, :2] / z.clamp_min(1e-8).unsqueeze(1)
    if k1 != 0.0:
        r2 = (xy * xy).sum(dim=1, keepdim=True)
        xy = xy * (1.0 + float(k1) * r2)
    uv = torch.stack([K[0, 0] * xy[:, 0] + K[0, 2], K[1, 1] * xy[:, 1] + K[1, 2]], dim=1)
    valid = valid & torch.isfinite(uv).all(dim=1)
    return uv, valid


def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    hp = np.concatenate([pts, ones], axis=1) @ H.T
    return hp[:, :2] / np.maximum(hp[:, 2:3], 1e-12)


def infinite_homography(K: np.ndarray, source_Rcw: np.ndarray, target_Rcw: np.ndarray) -> np.ndarray:
    """Homography induced by pure camera rotation, as in paper Section 4.2.1."""
    H = K @ target_Rcw @ source_Rcw.T @ np.linalg.inv(K)
    return H / H[2, 2]
