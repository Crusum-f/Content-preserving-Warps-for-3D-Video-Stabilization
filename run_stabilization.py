from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

from cp4stabilizer.colmap import Camera, ImageRecord, read_reconstruction
from cp4stabilizer.geometry import apply_homography, infinite_homography, project_points_torch
from cp4stabilizer.paths import make_paper_path, source_poses
from cp4stabilizer.warp import estimate_homography, warp_frame


@dataclass
class ProcessingDomain:
    camera_by_id: dict[int, Camera]
    rectify: bool
    label: str


@dataclass
class PrewarpResult:
    image: np.ndarray
    mask: np.ndarray
    source_points: np.ndarray
    target_points: np.ndarray
    weights: np.ndarray
    homography: np.ndarray | None
    mode: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SIGGRAPH 2009 content-preserving 3D video stabilization.")
    p.add_argument("--sfm", default="input/sfm/video8", help="Directory containing COLMAP cameras/images/points3D txt files.")
    p.add_argument("--frames", default="input/frames/video8", help="Directory containing source frames.")
    p.add_argument("--output", default="output/stabilized", help="Output frame directory.")
    p.add_argument("--video", default=None, help="Optional mp4 path to write.")
    p.add_argument("--path-mode", choices=["original", "smooth", "constant", "linear", "quadratic", "parabolic"], default="smooth")
    p.add_argument("--rotation-mode", choices=["original", "smooth", "constant", "linear", "quadratic", "parabolic"], default=None)
    p.add_argument("--smooth-sigma", type=float, default=24.0)
    p.add_argument("--grid-cols", type=int, default=64)
    p.add_argument("--grid-rows", type=int, default=36)
    p.add_argument("--alpha", type=float, default=20.0)
    p.add_argument("--anchor-weight", type=float, default=0.1, help="Numerical gauge that weakly anchors unconstrained mesh modes to the pre-warp grid.")
    p.add_argument("--min-track", type=int, default=20)
    p.add_argument("--fade", type=int, default=50)
    p.add_argument(
        "--min-temporal-weight",
        type=float,
        default=0.0,
        help="Floor nonzero temporal track weights; useful near sequence endpoints where --fade would otherwise remove all constraints.",
    )
    p.add_argument("--max-points", type=int, default=0, help="Cap per-frame constraints for speed; 0 keeps all, matching the paper.")
    p.add_argument("--prewarp", choices=["none", "infinite", "homography", "general"], default="general")
    p.add_argument("--rotation-fit", choices=["projection", "poly"], default="projection")
    p.add_argument("--rotation-fit-max-observations", type=int, default=0, help="0 uses all observations in Equation 4.")
    p.add_argument("--rotation-fit-max-nfev", type=int, default=100)
    p.add_argument("--crop", choices=["common", "none"], default="common")
    p.add_argument("--use-radial-distortion", action="store_true", help="Apply COLMAP radial distortion during target projection.")
    p.add_argument("--rectify-domain", choices=["auto", "on", "off"], default="auto", help="Undistort frames and constraints into a pinhole working domain.")
    p.add_argument("--rectify-alpha", type=float, default=0.0, help="OpenCV undistortion alpha used when --rectify-domain is on/auto.")
    p.add_argument("--source-points", choices=["observed", "projected"], default="observed", help="Use tracked image observations or source-camera reprojections for P-hat.")
    p.add_argument("--homography-corner-limit", type=float, default=4.0, help="Fallback to no prewarp if H sends corners outside this many frame widths/heights.")
    p.add_argument("--homography-ransac-threshold", type=float, default=0.0, help="RANSAC reprojection threshold in pixels for homography prewarp; 0 disables RANSAC.")
    p.add_argument("--homography-min-area-scale", type=float, default=0.85, help="Reject homography prewarps whose corner bounding-box area falls below this frame-area ratio.")
    p.add_argument("--homography-max-area-scale", type=float, default=2.0, help="Reject homography prewarps whose corner bounding-box area exceeds this frame-area ratio.")
    p.add_argument("--homography-min-side-scale", type=float, default=0.8, help="Reject homography prewarps whose corner bounding-box width or height falls below this frame-size ratio.")
    p.add_argument("--homography-max-side-scale", type=float, default=2.0, help="Reject homography prewarps whose corner bounding-box width or height exceeds this frame-size ratio.")
    p.add_argument("--bad-homography-fallback", choices=["none", "infinite"], default="infinite", help="Prewarp fallback used when a fitted homography is degenerate.")
    p.add_argument("--max-homography-residual", type=float, default=0.0, help="Drop mesh constraints whose post-prewarp target displacement exceeds this many pixels; 0 disables.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None, help="Exclusive frame index.")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device for PyTorch projection.")
    p.add_argument("--save-warp", action="store_true", help="Save per-frame warp parameters for point reprojection.")
    return p.parse_args()


def temporal_weights(
    point_ids: np.ndarray,
    image_index: int,
    track_lens: np.ndarray,
    first_frames: np.ndarray,
    last_frames: np.ndarray,
    fade: int,
    min_temporal_weight: float = 0.0,
) -> np.ndarray:
    ids = np.clip(point_ids, 0, len(track_lens) - 1)
    first = first_frames[ids].astype(np.float64)
    last = last_frames[ids].astype(np.float64)
    known = (first >= 0) & (last >= first)
    if fade <= 0:
        return np.ones_like(first)
    left = (image_index - first) / float(fade)
    right = (last - image_index) / float(fade)
    weights = np.minimum(np.minimum(left, right), 1.0)
    fallback = np.clip(track_lens[ids].astype(np.float64) / (2.0 * fade), 0.0, 1.0)
    weights = np.where(known, weights, fallback)
    weights = np.clip(weights, 0.0, 1.0)
    if min_temporal_weight > 0:
        weights = np.where(weights > 0, np.maximum(weights, min_temporal_weight), float(min_temporal_weight))
    return np.clip(weights, 0.0, 1.0)


def frame_constraints(
    image_record: ImageRecord,
    source_camera: Camera,
    working_camera: Camera,
    source_pose,
    desired_pose,
    xyz_by_id: np.ndarray,
    track_lens: np.ndarray,
    first_frames: np.ndarray,
    last_frames: np.ndarray,
    min_track: int,
    fade: int,
    min_temporal_weight: float,
    max_points: int,
    frame_idx: int,
    device: torch.device,
    use_radial_distortion: bool,
    rectify_observations: bool,
    source_points_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_ids = image_record.point3d_ids
    valid = point_ids >= 0
    valid &= point_ids < len(track_lens)
    valid &= track_lens[np.clip(point_ids, 0, len(track_lens) - 1)] >= min_track
    point_ids = point_ids[valid]
    observed_points = image_record.xys[valid]
    if len(point_ids) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,))

    xyz = xyz_by_id[point_ids]
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    observed_points = observed_points[finite]
    point_ids = point_ids[finite]
    with torch.no_grad():
        xyz_t = torch.as_tensor(xyz, dtype=torch.float64, device=device)
        K_t = torch.as_tensor(working_camera.K, dtype=torch.float64, device=device)
        source_R_t = torch.as_tensor(source_pose.Rcw, dtype=torch.float64, device=device)
        source_t_t = torch.as_tensor(source_pose.tvec, dtype=torch.float64, device=device)
        R_t = torch.as_tensor(desired_pose.Rcw, dtype=torch.float64, device=device)
        t_t = torch.as_tensor(desired_pose.tvec, dtype=torch.float64, device=device)
        k1 = working_camera.radial_k1 if use_radial_distortion else 0.0
        target_t, valid_t = project_points_torch(xyz_t, K_t, R_t, t_t, k1)
        target_points = target_t.cpu().numpy()
        proj_valid = valid_t.cpu().numpy()
        if source_points_mode == "projected":
            source_t, source_valid_t = project_points_torch(xyz_t, K_t, source_R_t, source_t_t, k1)
            source_points = source_t.cpu().numpy()
            proj_valid &= source_valid_t.cpu().numpy()
        elif source_points_mode == "observed":
            source_points = (
                source_camera.undistort_points(observed_points, working_camera)
                if rectify_observations
                else observed_points.copy()
            )
        else:
            raise ValueError(f"Unknown source point mode: {source_points_mode}")
    source_points = source_points[proj_valid]
    target_points = target_points[proj_valid]
    point_ids = point_ids[proj_valid]
    weights = temporal_weights(point_ids, frame_idx, track_lens, first_frames, last_frames, fade, min_temporal_weight)

    h, w = working_camera.height, working_camera.width
    inside = (
        (source_points[:, 0] >= 0)
        & (source_points[:, 0] < w)
        & (source_points[:, 1] >= 0)
        & (source_points[:, 1] < h)
        & np.isfinite(target_points).all(axis=1)
    )
    source_points = source_points[inside]
    target_points = target_points[inside]
    weights = weights[inside]

    active = weights > 0
    source_points = source_points[active]
    target_points = target_points[active]
    weights = weights[active]

    if max_points > 0 and len(weights) > max_points:
        rng = np.random.default_rng(frame_idx)
        if weights.sum() > 1e-12:
            prob = weights / weights.sum()
        else:
            prob = None
        keep = rng.choice(len(weights), size=max_points, replace=False, p=prob)
        source_points = source_points[keep]
        target_points = target_points[keep]
        weights = weights[keep]
    return source_points, target_points, weights


def build_frame_lookup(frames_dir: Path) -> tuple[dict[str, Path], dict[tuple[int, str], Path]]:
    by_name: dict[str, Path] = {}
    by_numeric_stem: dict[tuple[int, str], Path] = {}
    ambiguous: set[tuple[int, str]] = set()

    for path in frames_dir.iterdir():
        if not path.is_file():
            continue
        by_name[path.name] = path
        if path.stem.isdigit():
            key = (int(path.stem), path.suffix.lower())
            if key in by_numeric_stem:
                ambiguous.add(key)
            else:
                by_numeric_stem[key] = path

    for key in ambiguous:
        by_numeric_stem.pop(key, None)
    return by_name, by_numeric_stem


def resolve_frame_path(
    frames_dir: Path,
    image_name: str,
    by_name: dict[str, Path],
    by_numeric_stem: dict[tuple[int, str], Path],
) -> Path:
    exact = frames_dir / image_name
    if exact.exists():
        return exact

    name_path = Path(image_name)
    direct = by_name.get(name_path.name)
    if direct is not None:
        return direct

    if name_path.stem.isdigit():
        numeric = by_numeric_stem.get((int(name_path.stem), name_path.suffix.lower()))
        if numeric is not None:
            return numeric

    return exact


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    recon = read_reconstruction(args.sfm)
    images = recon.images
    domain = build_processing_domain(recon.cameras, args.rectify_domain, args.rectify_alpha)
    desired = make_paper_path(
        images,
        domain.camera_by_id,
        recon.point_xyz_by_id,
        recon.point_track_len_by_id,
        mode=args.path_mode,
        rotation_mode=args.rotation_mode,
        min_track=args.min_track,
        smooth_sigma=args.smooth_sigma,
        rotation_fit=args.rotation_fit,
        rotation_fit_max_observations=args.rotation_fit_max_observations,
        max_nfev=args.rotation_fit_max_nfev,
        use_radial_distortion=args.use_radial_distortion and not domain.rectify,
        observation_cameras=recon.cameras,
        rectify_observations=domain.rectify,
    )
    source = source_poses(images)

    frames_dir = Path(args.frames)
    frame_by_name, frame_by_numeric_stem = build_frame_lookup(frames_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    start = max(args.start, 0)
    end = len(images) if args.end is None else min(args.end, len(images))
    output_paths: list[Path] = []
    common_mask = None

    for idx in tqdm(range(start, end), desc="stabilizing"):
        imrec = images[idx]
        source_camera = recon.cameras[imrec.camera_id]
        working_camera = domain.camera_by_id[imrec.camera_id]
        frame_path = resolve_frame_path(frames_dir, imrec.name, frame_by_name, frame_by_numeric_stem)
        raw_image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if raw_image is None:
            raise FileNotFoundError(frame_path)
        image = source_camera.undistort_image(raw_image, working_camera) if domain.rectify else raw_image
        src, dst, weights = frame_constraints(
            imrec,
            source_camera,
            working_camera,
            source[idx],
            desired[idx],
            recon.point_xyz_by_id,
            recon.point_track_len_by_id,
            recon.point_first_frame_by_id,
            recon.point_last_frame_by_id,
            args.min_track,
            args.fade,
            args.min_temporal_weight,
            args.max_points,
            idx,
            device,
            args.use_radial_distortion and not domain.rectify,
            domain.rectify,
            args.source_points,
        )
        prewarp = prepare_prewarp(
            image,
            src,
            dst,
            weights,
            args.prewarp,
            working_camera,
            source[idx].Rcw,
            desired[idx].Rcw,
            args.homography_corner_limit,
            args.homography_ransac_threshold,
            args.homography_min_area_scale,
            args.homography_max_area_scale,
            args.homography_min_side_scale,
            args.homography_max_side_scale,
            args.bad_homography_fallback,
        )
        if prewarp.homography is not None and args.max_homography_residual > 0:
            pre_src, pre_dst, pre_weights = filter_homography_residuals(
                prewarp.source_points,
                prewarp.target_points,
                prewarp.weights,
                np.eye(3, dtype=np.float64),
                args.max_homography_residual,
            )
            prewarp.source_points = pre_src
            prewarp.target_points = pre_dst
            prewarp.weights = pre_weights
        result = warp_frame(
            prewarp.image,
            prewarp.source_points,
            prewarp.target_points,
            prewarp.weights,
            cols=args.grid_cols,
            rows=args.grid_rows,
            alpha=args.alpha,
            anchor_weight=args.anchor_weight,
            homography=None,
            input_mask=prewarp.mask,
        )
        out_path = output_dir / imrec.name
        mask_path = mask_dir / imrec.name
        cv2.imwrite(str(out_path), result.image)
        cv2.imwrite(str(mask_path), result.mask)
        output_paths.append(out_path)
        valid_mask = result.mask > 0
        common_mask = valid_mask if common_mask is None else (common_mask & valid_mask)

        if args.save_warp:
            warp_dir = output_dir / "warps"
            warp_dir.mkdir(parents=True, exist_ok=True)
            H = prewarp.homography if prewarp.homography is not None else np.eye(3, dtype=np.float64)
            np.savez_compressed(
                warp_dir / f"{Path(imrec.name).stem}.npz",
                homography=H,
                source_vertices=result.source_vertices,
                output_vertices=result.output_vertices,
                cols=np.int32(args.grid_cols),
                rows=np.int32(args.grid_rows),
                width=np.int32(working_camera.width),
                height=np.int32(working_camera.height),
            )

    if args.crop == "common" and common_mask is not None:
        crop_box = mask_largest_valid_rect(common_mask)
        if crop_box is not None:
            crop_outputs(output_paths, mask_dir, crop_box)
            import json
            (output_dir / "crop_box.json").write_text(
                json.dumps(list(crop_box)), encoding="utf-8"
            )

    if args.video:
        write_video(output_paths, args.video, args.fps)


def build_processing_domain(cameras: dict[int, Camera], mode: str, alpha: float) -> ProcessingDomain:
    if mode == "off":
        return ProcessingDomain(camera_by_id=dict(cameras), rectify=False, label="original")

    should_rectify = mode == "on" or any(camera.is_distorted() for camera in cameras.values())
    if not should_rectify:
        return ProcessingDomain(camera_by_id=dict(cameras), rectify=False, label="original")

    if len(cameras) == 1:
        camera_id, camera = next(iter(cameras.items()))
        return ProcessingDomain(
            camera_by_id={camera_id: camera.optimal_undistorted_camera(alpha)},
            rectify=True,
            label="shared-rectified",
        )

    width = next(iter(cameras.values())).width
    height = next(iter(cameras.values())).height
    fx = float(np.median([camera.K[0, 0] for camera in cameras.values()]))
    fy = float(np.median([camera.K[1, 1] for camera in cameras.values()]))
    cx = float(np.median([camera.K[0, 2] for camera in cameras.values()]))
    cy = float(np.median([camera.K[1, 2] for camera in cameras.values()]))
    working = {
        camera_id: Camera(camera_id, "PINHOLE", width, height, np.asarray([fx, fy, cx, cy], dtype=np.float64))
        for camera_id in cameras
    }
    return ProcessingDomain(camera_by_id=working, rectify=True, label="median-rectified")


def prepare_prewarp(
    image: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    weights: np.ndarray,
    mode: str,
    camera: Camera,
    source_Rcw: np.ndarray,
    target_Rcw: np.ndarray,
    corner_limit: float,
    ransac_threshold: float,
    min_area_scale: float,
    max_area_scale: float,
    min_side_scale: float,
    max_side_scale: float,
    bad_homography_fallback: str,
) -> PrewarpResult:
    h, w = image.shape[:2]
    full_mask = np.full((h, w), 255, dtype=np.uint8)
    H = make_prewarp_homography(mode, source_points, target_points, weights, camera.K, source_Rcw, target_Rcw, ransac_threshold)
    out_mode = mode
    if H is not None and not homography_has_reasonable_corners(
        H,
        w,
        h,
        corner_limit,
        min_area_scale,
        max_area_scale,
        min_side_scale,
        max_side_scale,
    ):
        H = fallback_homography(
            bad_homography_fallback,
            camera.K,
            source_Rcw,
            target_Rcw,
            w,
            h,
            corner_limit,
            min_area_scale,
            max_area_scale,
            min_side_scale,
            max_side_scale,
        )
        out_mode = f"fallback-{bad_homography_fallback}" if H is not None else "fallback-none"

    if H is None:
        return PrewarpResult(
            image=image,
            mask=full_mask,
            source_points=source_points,
            target_points=target_points,
            weights=weights,
            homography=None,
            mode="none" if mode == "none" else out_mode,
        )

    prewarped_image = cv2.warpPerspective(
        image,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    prewarped_mask = cv2.warpPerspective(
        full_mask,
        H,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    prewarped_source = apply_homography(source_points, H) if len(source_points) else source_points
    valid = (
        np.isfinite(prewarped_source).all(axis=1)
        & (prewarped_source[:, 0] >= 0)
        & (prewarped_source[:, 0] < w)
        & (prewarped_source[:, 1] >= 0)
        & (prewarped_source[:, 1] < h)
    )
    return PrewarpResult(
        image=prewarped_image,
        mask=prewarped_mask,
        source_points=prewarped_source[valid],
        target_points=target_points[valid],
        weights=weights[valid],
        homography=H,
        mode=out_mode,
    )


def homography_has_reasonable_corners(
    H: np.ndarray,
    width: int,
    height: int,
    limit_scale: float,
    min_area_scale: float,
    max_area_scale: float,
    min_side_scale: float,
    max_side_scale: float,
) -> bool:
    if H is None or not np.all(np.isfinite(H)):
        return False
    corners = np.array([[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]])
    warped = apply_homography(corners, H)
    if not np.all(np.isfinite(warped)):
        return False
    limit = float(limit_scale) * max(width, height)
    if not np.all(np.abs(warped) < limit):
        return False
    bbox_w = float(warped[:, 0].max() - warped[:, 0].min())
    bbox_h = float(warped[:, 1].max() - warped[:, 1].min())
    if bbox_w <= 0 or bbox_h <= 0:
        return False
    area_scale = (bbox_w * bbox_h) / float(width * height)
    if area_scale < min_area_scale or area_scale > max_area_scale:
        return False
    if bbox_w < min_side_scale * width or bbox_h < min_side_scale * height:
        return False
    if bbox_w > max_side_scale * width or bbox_h > max_side_scale * height:
        return False
    return bool(np.linalg.det(H[:2, :2]) > 0)


def fallback_homography(
    fallback: str,
    K: np.ndarray,
    source_Rcw: np.ndarray,
    target_Rcw: np.ndarray,
    width: int,
    height: int,
    corner_limit: float,
    min_area_scale: float,
    max_area_scale: float,
    min_side_scale: float,
    max_side_scale: float,
) -> np.ndarray | None:
    if fallback == "none":
        return None
    if fallback != "infinite":
        raise ValueError(f"Unknown bad homography fallback: {fallback}")
    H = infinite_homography(K, source_Rcw, target_Rcw)
    if homography_has_reasonable_corners(
        H,
        width,
        height,
        corner_limit,
        min_area_scale,
        max_area_scale,
        min_side_scale,
        max_side_scale,
    ):
        return H
    return None


def make_prewarp_homography(
    mode: str,
    source_points: np.ndarray,
    target_points: np.ndarray,
    weights: np.ndarray,
    K: np.ndarray,
    source_Rcw: np.ndarray,
    target_Rcw: np.ndarray,
    ransac_threshold: float,
) -> np.ndarray | None:
    if mode == "none":
        return None
    if mode == "infinite":
        return infinite_homography(K, source_Rcw, target_Rcw)
    if mode in {"homography", "general"}:
        if int((weights > 0).sum()) < 4:
            return None
        return estimate_homography(source_points, target_points, weights, ransac_max_error=ransac_threshold)
    raise ValueError(f"Unknown prewarp mode: {mode}")


def filter_homography_residuals(
    source_points: np.ndarray,
    target_points: np.ndarray,
    weights: np.ndarray,
    H: np.ndarray,
    max_residual: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from cp4stabilizer.geometry import apply_homography

    predicted = apply_homography(source_points, H)
    residual = np.linalg.norm(predicted - target_points, axis=1)
    keep = (weights <= 0) | (residual <= max_residual)
    return source_points[keep], target_points[keep], weights[keep]


def mask_largest_valid_rect(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Largest all-valid axis-aligned rectangle inside the common output mask."""
    valid = mask.astype(bool)
    if not valid.any():
        return None
    heights = np.zeros(valid.shape[1], dtype=np.int32)
    best_area = 0
    best = None
    for y, row in enumerate(valid):
        heights = np.where(row, heights + 1, 0)
        stack: list[int] = []
        for x in range(valid.shape[1] + 1):
            current = heights[x] if x < valid.shape[1] else 0
            while stack and heights[stack[-1]] > current:
                top = stack.pop()
                height = int(heights[top])
                left = stack[-1] + 1 if stack else 0
                right = x
                area = height * (right - left)
                if area > best_area:
                    best_area = area
                    best = (left, y - height + 1, right, y + 1)
            stack.append(x)
    if best is None:
        return None
    x0, y0, x1, y1 = best
    if (x1 - x0) % 2:
        x1 -= 1
    if (y1 - y0) % 2:
        y1 -= 1
    if x1 <= x0 or y1 <= y0:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def crop_outputs(output_paths: list[Path], mask_dir: Path, crop_box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = crop_box
    for out_path in output_paths:
        image = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
        if image is not None:
            cv2.imwrite(str(out_path), image[y0:y1, x0:x1])
        mask_path = mask_dir / out_path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            cv2.imwrite(str(mask_path), mask[y0:y1, x0:x1])


def write_video(output_paths: list[Path], video_path: str, fps: float) -> None:
    path = Path(video_path)
    if not path.suffix:
        path = path.with_suffix(".mp4")
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, macro_block_size=1) as writer:
        for out_path in output_paths:
            image = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            h, w = image.shape[:2]
            image = image[: h - (h % 2), : w - (w % 2)]
            writer.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    main()
