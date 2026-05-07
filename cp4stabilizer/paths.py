from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from .colmap import Camera, ImageRecord
from .geometry import camera_center_from_extrinsics, project_points_numpy, qvec_to_rotmat, tvec_from_center


@dataclass
class CameraPose:
    Rcw: np.ndarray
    tvec: np.ndarray
    center: np.ndarray


def source_poses(images: list[ImageRecord]) -> list[CameraPose]:
    poses: list[CameraPose] = []
    for im in images:
        Rcw = qvec_to_rotmat(im.qvec)
        center = camera_center_from_extrinsics(Rcw, im.tvec)
        poses.append(CameraPose(Rcw=Rcw, tvec=im.tvec.copy(), center=center))
    return poses


def _fit_poly(values: np.ndarray, degree: int) -> np.ndarray:
    n = len(values)
    t = np.linspace(-1.0, 1.0, n)
    degree = min(degree, n - 1)
    out = np.empty_like(values)
    for d in range(values.shape[1]):
        coeff = np.polyfit(t, values[:, d], degree)
        out[:, d] = np.polyval(coeff, t)
    return out


def _fit_poly_coefficients(values: np.ndarray, degree: int) -> np.ndarray:
    n = len(values)
    t = np.linspace(-1.0, 1.0, n)
    degree = min(degree, n - 1)
    coeff = np.empty((degree + 1, values.shape[1]), dtype=np.float64)
    for d in range(values.shape[1]):
        coeff[:, d] = np.polyfit(t, values[:, d], degree)
    return coeff


def _eval_poly_coefficients(coeff: np.ndarray, count: int) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, count)
    out = np.empty((count, coeff.shape[1]), dtype=np.float64)
    for d in range(coeff.shape[1]):
        out[:, d] = np.polyval(coeff[:, d], t)
    return out


def _smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return values.copy()
    return gaussian_filter1d(values, sigma=sigma, axis=0, mode="nearest")


def _smooth_rotations_local(rotations: Rotation, sigma: float) -> np.ndarray:
    """Low-pass filter rotations in a per-frame tangent space.

    This follows the spirit of Lee and Shin's orientation filtering used by
    the paper: each output orientation is formed by averaging neighboring
    rotations after mapping them into the local coordinate frame of the
    current sample.
    """
    if sigma <= 0:
        return rotations.as_matrix()
    count = len(rotations)
    radius = max(1, int(np.ceil(4.0 * sigma)))
    out = np.empty((count, 3, 3), dtype=np.float64)
    for idx in range(count):
        lo = max(0, idx - radius)
        hi = min(count, idx + radius + 1)
        offsets = np.arange(lo, hi, dtype=np.float64) - float(idx)
        weights = np.exp(-0.5 * (offsets / sigma) ** 2)
        weights /= weights.sum()
        local = rotations[idx].inv() * rotations[lo:hi]
        mean_rotvec = np.sum(local.as_rotvec() * weights[:, None], axis=0)
        out[idx] = (rotations[idx] * Rotation.from_rotvec(mean_rotvec)).as_matrix()
    return out


def make_desired_path(
    images: list[ImageRecord],
    mode: str = "linear",
    rotation_mode: str | None = None,
    smooth_sigma: float = 24.0,
) -> list[CameraPose]:
    """Create the target path from the recovered COLMAP poses.

    The paper fits user-selected constant/linear/quadratic camera models.
    Here the position is fit in Euclidean camera-center space and the
    orientation is fit in exponential coordinates, matching the paper's
    practical parameterization.
    """
    src = source_poses(images)
    centers = np.stack([p.center for p in src], axis=0)
    rotations = Rotation.from_matrix(np.stack([p.Rcw for p in src], axis=0))
    rotvecs = np.unwrap(rotations.as_rotvec(), axis=0)

    rotation_mode = rotation_mode or mode
    pos_values = _path_values(centers, mode, smooth_sigma)
    if rotation_mode == "smooth":
        desired_rots = _smooth_rotations_local(rotations, smooth_sigma)
    else:
        rot_values = _path_values(rotvecs, rotation_mode, smooth_sigma)
        desired_rots = Rotation.from_rotvec(rot_values).as_matrix()
    poses: list[CameraPose] = []
    for Rcw, center in zip(desired_rots, pos_values):
        poses.append(CameraPose(Rcw=Rcw, tvec=tvec_from_center(Rcw, center), center=center))
    return poses


def make_paper_path(
    images: list[ImageRecord],
    cameras: dict[int, Camera],
    xyz_by_id: np.ndarray,
    track_lens_by_id: np.ndarray,
    mode: str = "linear",
    rotation_mode: str | None = None,
    min_track: int = 20,
    smooth_sigma: float = 24.0,
    rotation_fit: str = "projection",
    rotation_fit_max_observations: int = 0,
    max_nfev: int = 100,
    use_radial_distortion: bool = False,
    observation_cameras: dict[int, Camera] | None = None,
    rectify_observations: bool = False,
) -> list[CameraPose]:
    """Camera path fitting matching Section 4.3.

    Position is fit by least squares. For constant/linear/quadratic rotation
    models, the paper chooses rotation parameters by minimizing feature
    projection disparity (Equation 4); ``rotation_fit='projection'`` performs
    that nonlinear least-squares refinement.
    """
    src = source_poses(images)
    centers = np.stack([p.center for p in src], axis=0)
    rotations = Rotation.from_matrix(np.stack([p.Rcw for p in src], axis=0))
    rotvecs = np.unwrap(rotations.as_rotvec(), axis=0)

    rotation_mode = rotation_mode or mode
    pos_values = _path_values(centers, mode, smooth_sigma)
    if rotation_mode == "smooth":
        desired_rots = _smooth_rotations_local(rotations, smooth_sigma)
        rot_values = None
    else:
        rot_values = _path_values(rotvecs, rotation_mode, smooth_sigma)
        desired_rots = Rotation.from_rotvec(rot_values).as_matrix()

    if rotation_fit == "projection" and rotation_mode in {"constant", "linear", "quadratic", "parabolic"}:
        degree = _mode_degree(rotation_mode)
        coeff0 = _fit_poly_coefficients(rotvecs, degree)
        coeff = _fit_rotation_by_projection(
            coeff0,
            images,
            cameras,
            xyz_by_id,
            track_lens_by_id,
            pos_values,
            min_track=min_track,
            max_observations=rotation_fit_max_observations,
            max_nfev=max_nfev,
            use_radial_distortion=use_radial_distortion,
            observation_cameras=observation_cameras,
            rectify_observations=rectify_observations,
        )
        rot_values = _eval_poly_coefficients(coeff, len(images))
        desired_rots = Rotation.from_rotvec(rot_values).as_matrix()

    return [CameraPose(Rcw=Rcw, tvec=tvec_from_center(Rcw, center), center=center) for Rcw, center in zip(desired_rots, pos_values)]


def _path_values(values: np.ndarray, mode: str, smooth_sigma: float) -> np.ndarray:
    if mode == "original":
        return values.copy()
    if mode == "smooth":
        return _smooth(values, smooth_sigma)
    if mode == "constant":
        return np.repeat(values.mean(axis=0, keepdims=True), len(values), axis=0)
    if mode == "linear":
        return _fit_poly(values, 1)
    if mode in {"quadratic", "parabolic"}:
        return _fit_poly(values, 2)
    raise ValueError(f"Unknown path mode: {mode}")


def _mode_degree(mode: str) -> int:
    if mode == "constant":
        return 0
    if mode == "linear":
        return 1
    if mode in {"quadratic", "parabolic"}:
        return 2
    raise ValueError(f"Projection rotation fitting requires constant/linear/quadratic, got {mode}")


def _fit_rotation_by_projection(
    coeff0: np.ndarray,
    images: list[ImageRecord],
    cameras: dict[int, Camera],
    xyz_by_id: np.ndarray,
    track_lens_by_id: np.ndarray,
    centers: np.ndarray,
    min_track: int,
    max_observations: int,
    max_nfev: int,
    use_radial_distortion: bool,
    observation_cameras: dict[int, Camera] | None,
    rectify_observations: bool,
) -> np.ndarray:
    obs_frames, obs_xy, obs_xyz = _collect_projection_observations(
        images,
        cameras,
        xyz_by_id,
        track_lens_by_id,
        min_track=min_track,
        max_observations=max_observations,
        observation_cameras=observation_cameras,
        rectify_observations=rectify_observations,
    )
    if len(obs_frames) == 0:
        return coeff0

    frame_to_obs = [np.flatnonzero(obs_frames == idx) for idx in range(len(images))]
    degree = coeff0.shape[0] - 1
    t_values = np.linspace(-1.0, 1.0, len(images))

    def unpack(params: np.ndarray) -> np.ndarray:
        return params.reshape(degree + 1, 3)

    def residual(params: np.ndarray) -> np.ndarray:
        coeff = unpack(params)
        rotvecs = np.empty((len(images), 3), dtype=np.float64)
        for d in range(3):
            rotvecs[:, d] = np.polyval(coeff[:, d], t_values)
        rotations = Rotation.from_rotvec(rotvecs).as_matrix()
        out = np.empty_like(obs_xy)
        for frame_idx, obs_idx in enumerate(frame_to_obs):
            if len(obs_idx) == 0:
                continue
            camera = cameras[images[frame_idx].camera_id]
            Rcw = rotations[frame_idx]
            tvec = tvec_from_center(Rcw, centers[frame_idx])
            uv, valid = project_points_numpy(obs_xyz[obs_idx], camera, Rcw, tvec, apply_distortion=use_radial_distortion)
            uv[~valid] = 1e6
            out[obs_idx] = uv
        return (out - obs_xy).reshape(-1)

    method = "lm" if len(obs_xy) * 2 > coeff0.size else "trf"
    result = least_squares(residual, coeff0.reshape(-1), method=method, max_nfev=max_nfev)
    return result.x.reshape(coeff0.shape)


def _collect_projection_observations(
    images: list[ImageRecord],
    cameras: dict[int, Camera],
    xyz_by_id: np.ndarray,
    track_lens_by_id: np.ndarray,
    min_track: int,
    max_observations: int,
    observation_cameras: dict[int, Camera] | None = None,
    rectify_observations: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = []
    xys = []
    xyzs = []
    for frame_idx, im in enumerate(images):
        ids = im.point3d_ids
        valid = ids >= 0
        valid &= ids < len(track_lens_by_id)
        valid &= track_lens_by_id[np.clip(ids, 0, len(track_lens_by_id) - 1)] >= min_track
        ids = ids[valid]
        if len(ids) == 0:
            continue
        xyz = xyz_by_id[ids]
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        xy = im.xys[valid][finite]
        if rectify_observations:
            if observation_cameras is None:
                raise ValueError("observation_cameras is required when rectify_observations=True")
            xy = observation_cameras[im.camera_id].undistort_points(xy, cameras[im.camera_id])
        frames.append(np.full((len(xy),), frame_idx, dtype=np.int32))
        xys.append(xy.astype(np.float64))
        xyzs.append(xyz.astype(np.float64))
    if not frames:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 2)), np.zeros((0, 3))

    obs_frames = np.concatenate(frames)
    obs_xy = np.concatenate(xys)
    obs_xyz = np.concatenate(xyzs)
    if max_observations > 0 and len(obs_frames) > max_observations:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(obs_frames), size=max_observations, replace=False)
        obs_frames = obs_frames[keep]
        obs_xy = obs_xy[keep]
        obs_xyz = obs_xyz[keep]
    return obs_frames, obs_xy, obs_xyz
