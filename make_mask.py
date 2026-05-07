#!/usr/bin/env python3
"""Generate COLMAP masks that ignore people in video frames.

The default paths match the video9/0005 example in this repository:

    python make_mask.py

COLMAP treats non-zero mask pixels as valid. This script therefore writes
single-channel PNG masks where background is white (255) and detected people
are black (0).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PERSON_CLASS_ID = 1  # COCO class id used by torchvision detection models.


@dataclass
class PersonSegmentation:
    person_mask: np.ndarray
    score: float
    count: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Automatically generate white-background / black-person masks for "
            "COLMAP feature extraction."
        )
    )
    p.add_argument("--frames", default="input/frames/video9/0005", help="Directory containing extracted video frames.")
    p.add_argument("--video", default="input/video/video9/0005.avi", help="Optional video used only when --extract-frames is set.")
    p.add_argument(
        "--output",
        default=None,
        help="Mask output directory. Defaults to input/masks/<same suffix as frames under input/frames>.",
    )
    p.add_argument(
        "--name-mode",
        default="both",
        choices=["same", "colmap", "both"],
        help="Mask filename style: same=0001.png, colmap=0001.png.png, both=writes both.",
    )
    p.add_argument("--extract-frames", action="store_true", help="Extract frames from --video into --frames before masking.")
    p.add_argument("--overwrite-frames", action="store_true", help="Overwrite existing extracted frames when --extract-frames is used.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Inference device.")
    p.add_argument("--score-threshold", type=float, default=0.55, help="Minimum person detection score.")
    p.add_argument("--mask-threshold", type=float, default=0.5, help="Per-pixel instance mask threshold.")
    p.add_argument("--min-area", type=int, default=1000, help="Drop detected person masks smaller than this many pixels.")
    p.add_argument("--box-pad", type=int, default=8, help="Expand detected person boxes by this many pixels when SAM refinement or --fill-box is used.")
    p.add_argument("--fill-box", action="store_true", help="Conservatively mask the whole expanded person bounding box.")
    p.add_argument("--dilate", type=int, default=13, help="Dilate person masks with this odd kernel size. Use 0 to disable.")
    p.add_argument("--close", type=int, default=11, help="Morphologically close person masks with this odd kernel size. Use 0 to disable.")
    p.add_argument("--batch-size", type=int, default=2, help="Number of frames processed per model forward pass.")
    p.add_argument("--limit", type=int, default=0, help="Process only the first N frames. Useful for smoke tests.")
    p.add_argument("--preview-dir", default="output/mask_preview_video9_0005", help="Directory for overlay previews. Use empty string to disable.")
    p.add_argument("--preview-every", type=int, default=60, help="Write an overlay preview every N frames, plus first and last.")
    p.add_argument("--inpaint-dir", default="", help="Optional output directory for frames with person regions inpainted.")
    p.add_argument("--inpaint-radius", type=float, default=5.0, help="OpenCV inpaint radius for --inpaint-dir.")
    p.add_argument("--sam-checkpoint", default="", help="Optional SAM checkpoint path for box-prompt mask refinement.")
    p.add_argument("--sam-model", default="vit_h", choices=["vit_h", "vit_l", "vit_b"], help="SAM model type.")
    return p.parse_args()


def default_mask_dir(frames_dir: Path) -> Path:
    parts = frames_dir.parts
    if "frames" in parts:
        idx = parts.index("frames")
        return Path(*parts[:idx], "masks", *parts[idx + 1 :])
    return frames_dir.parent / f"{frames_dir.name}_masks"


def odd_kernel(size: int) -> np.ndarray | None:
    if size <= 0:
        return None
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def list_frames(frames_dir: Path, limit: int = 0) -> list[Path]:
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frame directory does not exist: {frames_dir}")
    files = sorted(p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"No image frames found in: {frames_dir}")
    return files[:limit] if limit > 0 else files


def extract_frames(video_path: Path, frames_dir: Path, overwrite: bool = False) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = list(frames_dir.glob("*.png"))
    if existing and not overwrite:
        print(f"[extract] Skip: {frames_dir} already contains {len(existing)} PNG frames.")
        return

    if overwrite:
        for p in existing:
            p.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    idx = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(frames_dir / f"{idx:04d}.png"), frame)
        idx += 1
    cap.release()
    print(f"[extract] Wrote {idx - 1} frames to {frames_dir}")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_mask_rcnn(device: torch.device) -> torch.nn.Module:
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights)
    model.to(device)
    model.eval()
    return model


def load_sam_predictor(checkpoint: str, model_type: str, device: torch.device):
    if not checkpoint:
        return None
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint does not exist: {checkpoint_path}")
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise RuntimeError("segment_anything is not installed in this environment.") from exc

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device)
    return SamPredictor(sam)


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img


def batch_tensors(images_bgr: list[np.ndarray], device: torch.device) -> list[torch.Tensor]:
    tensors = []
    for img in images_bgr:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensors.append(F.to_tensor(rgb).to(device))
    return tensors


def box_to_int(box: np.ndarray, width: int, height: int, pad: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box.astype(float).tolist()
    x0 = max(0, int(np.floor(x0 - pad)))
    y0 = max(0, int(np.floor(y0 - pad)))
    x1 = min(width, int(np.ceil(x1 + pad)))
    y1 = min(height, int(np.ceil(y1 + pad)))
    return x0, y0, x1, y1


def refine_with_sam(
    predictor,
    image_bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    height: int,
    width: int,
) -> np.ndarray | None:
    if predictor is None or not boxes:
        return None

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    refined = np.zeros((height, width), dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        box = np.array([x0, y0, x1, y1], dtype=np.float32)
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        if len(masks) == 0:
            continue
        refined[masks[int(np.argmax(scores))]] = 255
    return refined


def postprocess_person_mask(person_mask: np.ndarray, close_size: int, dilate_size: int) -> np.ndarray:
    mask = person_mask.astype(np.uint8)
    close_kernel = odd_kernel(close_size)
    dilate_kernel = odd_kernel(dilate_size)
    if close_kernel is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    if dilate_kernel is not None:
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def segment_people(
    prediction: dict[str, torch.Tensor],
    image_bgr: np.ndarray,
    args: argparse.Namespace,
    sam_predictor=None,
) -> PersonSegmentation:
    height, width = image_bgr.shape[:2]
    labels = prediction["labels"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    masks = prediction["masks"].detach().cpu().numpy()
    boxes = prediction["boxes"].detach().cpu().numpy()

    person_mask = np.zeros((height, width), dtype=np.uint8)
    person_boxes: list[tuple[int, int, int, int]] = []
    best_score = 0.0
    count = 0

    for label, score, mask, box in zip(labels, scores, masks, boxes):
        if int(label) != PERSON_CLASS_ID or float(score) < args.score_threshold:
            continue
        instance = (mask[0] >= args.mask_threshold).astype(np.uint8) * 255
        if int((instance > 0).sum()) < args.min_area:
            continue

        x0, y0, x1, y1 = box_to_int(box, width, height, args.box_pad)
        if args.fill_box:
            instance[y0:y1, x0:x1] = 255
        person_mask = np.maximum(person_mask, instance)
        person_boxes.append((x0, y0, x1, y1))
        best_score = max(best_score, float(score))
        count += 1

    sam_mask = refine_with_sam(sam_predictor, image_bgr, person_boxes, height, width)
    if sam_mask is not None and np.any(sam_mask):
        person_mask = np.maximum(person_mask, sam_mask)

    person_mask = postprocess_person_mask(person_mask, args.close, args.dilate)
    return PersonSegmentation(person_mask=person_mask, score=best_score, count=count)


def colmap_mask_from_person(person_mask: np.ndarray) -> np.ndarray:
    mask = np.full(person_mask.shape, 255, dtype=np.uint8)
    mask[person_mask > 0] = 0
    return mask


def mask_output_paths(output_dir: Path, frame_path: Path, name_mode: str) -> list[Path]:
    paths: list[Path] = []
    if name_mode in {"same", "both"}:
        paths.append(output_dir / frame_path.name)
    if name_mode in {"colmap", "both"}:
        paths.append(output_dir / f"{frame_path.name}.png")
    return paths


def make_overlay(image_bgr: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    red = np.zeros_like(image_bgr)
    red[:, :, 2] = 255
    alpha = 0.45
    covered = person_mask > 0
    overlay[covered] = cv2.addWeighted(image_bgr, 1.0 - alpha, red, alpha, 0)[covered]
    return overlay


def should_write_preview(index: int, total: int, every: int) -> bool:
    if every <= 0:
        return index == 0 or index == total - 1
    return index == 0 or index == total - 1 or (index + 1) % every == 0


def run() -> int:
    args = parse_args()
    frames_dir = Path(args.frames)
    output_dir = Path(args.output) if args.output else default_mask_dir(frames_dir)
    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    inpaint_dir = Path(args.inpaint_dir) if args.inpaint_dir else None

    if args.extract_frames:
        extract_frames(Path(args.video), frames_dir, overwrite=args.overwrite_frames)

    frame_paths = list_frames(frames_dir, args.limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
    if inpaint_dir is not None:
        inpaint_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"[model] Loading Mask R-CNN on {device}...")
    model = load_mask_rcnn(device)
    sam_predictor = load_sam_predictor(args.sam_checkpoint, args.sam_model, device)
    if sam_predictor is not None:
        print(f"[model] SAM refinement enabled: {args.sam_model}")

    total_people = 0
    with torch.inference_mode():
        for start in range(0, len(frame_paths), args.batch_size):
            batch_paths = frame_paths[start : start + args.batch_size]
            images = [read_image(p) for p in batch_paths]
            predictions = model(batch_tensors(images, device))

            for local_idx, (path, img, prediction) in enumerate(zip(batch_paths, images, predictions)):
                index = start + local_idx
                seg = segment_people(prediction, img, args, sam_predictor=sam_predictor)
                mask = colmap_mask_from_person(seg.person_mask)
                written_paths = mask_output_paths(output_dir, path, args.name_mode)
                for mask_path in written_paths:
                    cv2.imwrite(str(mask_path), mask)

                if preview_dir is not None and should_write_preview(index, len(frame_paths), args.preview_every):
                    stem = path.stem
                    cv2.imwrite(str(preview_dir / f"{stem}_mask.png"), mask)
                    cv2.imwrite(str(preview_dir / f"{stem}_overlay.png"), make_overlay(img, seg.person_mask))

                if inpaint_dir is not None:
                    inpainted = cv2.inpaint(img, seg.person_mask, args.inpaint_radius, cv2.INPAINT_TELEA)
                    cv2.imwrite(str(inpaint_dir / path.name), inpainted)

                total_people += seg.count
                print(
                    f"[{index + 1:04d}/{len(frame_paths):04d}] {path.name}: "
                    f"people={seg.count} score={seg.score:.3f} -> {written_paths[0]}"
                )

    if args.name_mode == "both":
        print("[done] Wrote both same-name masks and COLMAP .png-appended masks.")
    print(f"[done] Wrote masks for {len(frame_paths)} frames to {output_dir}")
    print(f"[done] Total detected person instances: {total_people}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
