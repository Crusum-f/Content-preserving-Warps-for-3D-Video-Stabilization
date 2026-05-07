#!/usr/bin/env python3
"""Draw 3D point projections onto source frames, then run stabilization on them.

This bakes green dots into the source frames as pixels, so they go through
the exact same undistort+warp+crop transform as the image content.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from cp4stabilizer.colmap import read_reconstruction
from cp4stabilizer.geometry import project_points_numpy, qvec_to_rotmat


GREEN_BGR = (0, 255, 0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bake 3D point projections into frames and run stabilization.")
    p.add_argument("--sfm", default="input/sfm/video8")
    p.add_argument("--frames", default="input/frames/video8")
    p.add_argument("--output-frames", default="input/frames/video8_dotted")
    p.add_argument("--stabilized-output", default="output/stabilized_dotted")
    p.add_argument("--video-output", default="output/video8_stabilized_with_points.mp4")
    p.add_argument("--min-track", type=int, default=90, help="Minimum number of observations for a point to be shown.")
    p.add_argument("--max-points", type=int, default=50, help="Global top-N points by track length; each frame draws the visible subset.")
    p.add_argument("--radius", type=int, default=3)
    p.add_argument("--fps", type=float, default=30.0)
    return p.parse_args()


def select_top_point_ids(recon, min_track: int, max_points: int) -> np.ndarray:
    valid = np.flatnonzero(recon.point_track_len_by_id >= int(min_track))
    finite = np.isfinite(recon.point_xyz_by_id[valid]).all(axis=1)
    valid = valid[finite]
    if len(valid) <= max_points:
        return valid.astype(np.int64)
    track_len = recon.point_track_len_by_id[valid]
    order = np.argsort(-track_len)
    return valid[order[:max_points]].astype(np.int64)


def draw_green_points(frame: np.ndarray, points: np.ndarray, radius: int) -> np.ndarray:
    if len(points) == 0:
        return frame
    h, w = frame.shape[:2]
    inside = (
        (points[:, 0] >= 0.0) & (points[:, 0] < w)
        & (points[:, 1] >= 0.0) & (points[:, 1] < h)
        & np.isfinite(points).all(axis=1)
    )
    points = points[inside]
    if len(points) == 0:
        return frame
    for x, y in np.rint(points).astype(np.int32):
        cv2.circle(frame, (int(x), int(y)), int(radius), GREEN_BGR, thickness=-1, lineType=cv2.LINE_AA)
    return frame


def bake_frames(args: argparse.Namespace) -> None:
    recon = read_reconstruction(args.sfm)
    selected_ids = select_top_point_ids(recon, args.min_track, args.max_points)
    selected_mask = np.zeros((len(recon.point_track_len_by_id),), dtype=bool)
    selected_mask[selected_ids] = True
    print(f"Selected {len(selected_ids)} points (track_len>={args.min_track}, top {args.max_points})")

    frames_dir = Path(args.frames)
    out_dir = Path(args.output_frames)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for frame_idx in tqdm(range(len(recon.images)), desc="baking points"):
        img_rec = recon.images[frame_idx]
        src_path = frames_dir / img_rec.name
        frame = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(src_path)

        # Filter to valid point IDs visible in this frame
        point_ids = img_rec.point3d_ids
        valid = point_ids >= 0
        valid &= point_ids < len(recon.point_track_len_by_id)
        safe = np.clip(point_ids, 0, len(recon.point_track_len_by_id) - 1)
        valid &= selected_mask[safe]

        pids = point_ids[valid]
        if len(pids) == 0:
            cv2.imwrite(str(out_dir / img_rec.name), frame)
            continue

        xyz = recon.point_xyz_by_id[pids]
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        if len(xyz) == 0:
            cv2.imwrite(str(out_dir / img_rec.name), frame)
            continue

        camera = recon.cameras[img_rec.camera_id]
        Rcw = qvec_to_rotmat(img_rec.qvec)
        xy, valid_proj = project_points_numpy(xyz, camera, Rcw, img_rec.tvec, apply_distortion=True)
        xy = xy[valid_proj]
        if len(xy) == 0:
            cv2.imwrite(str(out_dir / img_rec.name), frame)
            continue

        dotted = draw_green_points(frame, xy, args.radius)
        cv2.imwrite(str(out_dir / img_rec.name), dotted)


def run_stabilization(args: argparse.Namespace) -> None:
    import subprocess
    import sys

    cmd = [
        sys.executable, "run_stabilization.py",
        "--sfm", args.sfm,
        "--frames", args.output_frames,
        "--output", args.stabilized_output,
        "--save-warp",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def make_video(args: argparse.Namespace) -> None:
    import imageio.v2 as imageio

    stab_dir = Path(args.stabilized_output)
    frames = sorted(
        [p for p in stab_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
        key=lambda p: p.name,
    )
    if not frames:
        print("No output frames found!")
        return

    out = Path(args.video_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(out), fps=args.fps, macro_block_size=1) as writer:
        for path in tqdm(frames, desc="writing video"):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            img = img[: h - (h % 2), : w - (w % 2)]
            writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    print(f"Saved: {out}")


def main() -> None:
    args = parse_args()
    print("=== Step 1: Bake 3D point projections into source frames ===")
    bake_frames(args)
    print(f"Baked {len(list(Path(args.output_frames).iterdir()))} frames to {args.output_frames}")

    print("\n=== Step 2: Run stabilization on dotted frames ===")
    run_stabilization(args)

    print("\n=== Step 3: Generate output video ===")
    make_video(args)


if __name__ == "__main__":
    main()
