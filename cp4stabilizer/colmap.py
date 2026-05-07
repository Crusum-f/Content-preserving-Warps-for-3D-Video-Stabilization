from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def K(self) -> np.ndarray:
        if self.model == "SIMPLE_RADIAL":
            f, cx, cy, _ = self.params
            return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        if self.model == "PINHOLE":
            fx, fy, cx, cy = self.params
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        raise ValueError(f"Unsupported camera model: {self.model}")

    @property
    def radial_k1(self) -> float:
        if self.model == "SIMPLE_RADIAL":
            return float(self.params[3])
        return 0.0

    def is_distorted(self) -> bool:
        return self.model == "SIMPLE_RADIAL" and abs(self.radial_k1) > 1e-12

    def distortion_coeffs(self) -> np.ndarray:
        coeffs = np.zeros((4,), dtype=np.float64)
        if self.model == "SIMPLE_RADIAL":
            coeffs[0] = self.radial_k1
        return coeffs

    def with_intrinsics(self, K: np.ndarray, model: str = "PINHOLE") -> "Camera":
        K = np.asarray(K, dtype=np.float64)
        if model != "PINHOLE":
            raise ValueError(f"Unsupported replacement camera model: {model}")
        params = np.asarray([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64)
        return Camera(self.camera_id, model, self.width, self.height, params)

    def optimal_undistorted_camera(self, alpha: float = 0.0) -> "Camera":
        if not self.is_distorted():
            return self.with_intrinsics(self.K, model="PINHOLE")
        new_K, _roi = cv2.getOptimalNewCameraMatrix(
            self.K,
            self.distortion_coeffs(),
            (self.width, self.height),
            float(alpha),
            (self.width, self.height),
        )
        return self.with_intrinsics(new_K, model="PINHOLE")

    def undistort_image(self, image: np.ndarray, new_camera: "Camera") -> np.ndarray:
        if not self.is_distorted():
            return image.copy()
        return cv2.undistort(image, self.K, self.distortion_coeffs(), None, new_camera.K)

    def undistort_points(self, points: np.ndarray, new_camera: "Camera") -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        if not self.is_distorted():
            return pts.reshape(-1, 2)
        undistorted = cv2.undistortPoints(pts, self.K, self.distortion_coeffs(), P=new_camera.K)
        return undistorted.reshape(-1, 2)


@dataclass
class ImageRecord:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3d_ids: np.ndarray


@dataclass
class Reconstruction:
    cameras: dict[int, Camera]
    images: list[ImageRecord]
    point_xyz_by_id: np.ndarray
    point_track_len_by_id: np.ndarray
    point_first_frame_by_id: np.ndarray
    point_last_frame_by_id: np.ndarray


def _read_data_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                yield stripped


def _natural_sort_key(value: str) -> tuple:
    parts = re.split(r"(\d+)", value)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def read_cameras(path: str | Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line in _read_data_lines(Path(path)):
        toks = line.split()
        camera_id = int(toks[0])
        model = toks[1]
        width = int(toks[2])
        height = int(toks[3])
        params = np.asarray([float(v) for v in toks[4:]], dtype=np.float64)
        cameras[camera_id] = Camera(camera_id, model, width, height, params)
    return cameras


def read_images(path: str | Path) -> list[ImageRecord]:
    def iter_image_records(path: Path):
        with path.open("r", encoding="utf-8") as f:
            lines = iter(f)
            for raw_header in lines:
                header = raw_header.strip()
                if not header or header.startswith("#"):
                    continue
                try:
                    pts_line = next(lines).strip()
                except StopIteration as exc:
                    raise ValueError(f"Missing POINTS2D line after image header in {path}: {header}") from exc
                yield header, pts_line

    images: list[ImageRecord] = []
    for header, pts_line in iter_image_records(Path(path)):
        h = header.split()
        image_id = int(h[0])
        qvec = np.asarray([float(v) for v in h[1:5]], dtype=np.float64)
        tvec = np.asarray([float(v) for v in h[5:8]], dtype=np.float64)
        camera_id = int(h[8])
        name = h[9]

        vals = pts_line.split()
        if vals:
            arr = np.asarray(vals, dtype=np.float64).reshape(-1, 3)
            xys = arr[:, :2].astype(np.float64)
            point3d_ids = arr[:, 2].astype(np.int64)
        else:
            xys = np.zeros((0, 2), dtype=np.float64)
            point3d_ids = np.zeros((0,), dtype=np.int64)
        images.append(ImageRecord(image_id, qvec, tvec, camera_id, name, xys, point3d_ids))
    return sorted(images, key=lambda im: _natural_sort_key(im.name))


def read_points3d(path: str | Path, image_id_to_index: dict[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw: list[tuple[int, np.ndarray, int, int, int]] = []
    max_id = 0
    for line in _read_data_lines(Path(path)):
        toks = line.split()
        point_id = int(toks[0])
        xyz = np.asarray([float(v) for v in toks[1:4]], dtype=np.float64)
        track_pairs = toks[8:]
        track_len = max(0, len(track_pairs) // 2)
        frame_indices = []
        for k in range(0, len(track_pairs), 2):
            frame_idx = image_id_to_index.get(int(track_pairs[k]))
            if frame_idx is not None:
                frame_indices.append(frame_idx)
        first = min(frame_indices) if frame_indices else -1
        last = max(frame_indices) if frame_indices else -1
        raw.append((point_id, xyz, track_len, first, last))
        max_id = max(max_id, point_id)

    xyz_by_id = np.full((max_id + 1, 3), np.nan, dtype=np.float64)
    track_len_by_id = np.zeros((max_id + 1,), dtype=np.int32)
    first_by_id = np.full((max_id + 1,), -1, dtype=np.int32)
    last_by_id = np.full((max_id + 1,), -1, dtype=np.int32)
    for point_id, xyz, track_len, first, last in raw:
        xyz_by_id[point_id] = xyz
        track_len_by_id[point_id] = track_len
        first_by_id[point_id] = first
        last_by_id[point_id] = last
    return xyz_by_id, track_len_by_id, first_by_id, last_by_id


def read_reconstruction(sfm_dir: str | Path) -> Reconstruction:
    sfm_dir = Path(sfm_dir)
    cameras = read_cameras(sfm_dir / "cameras.txt")
    images = read_images(sfm_dir / "images.txt")
    image_id_to_index = {im.image_id: idx for idx, im in enumerate(images)}
    xyz_by_id, track_len_by_id, first_by_id, last_by_id = read_points3d(sfm_dir / "points3D.txt", image_id_to_index)
    return Reconstruction(cameras, images, xyz_by_id, track_len_by_id, first_by_id, last_by_id)
