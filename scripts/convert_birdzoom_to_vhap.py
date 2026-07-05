#!/usr/bin/env python3
"""Convert BirdZoom captures into the VHAP dataset layout used by lumio_base.

The converter creates one dataset folder per BirdZoom dynamic sequence:
  - birdzoom_01
  - birdzoom_02
  - birdzoom_03

Each output folder contains:
  - images/
  - fg_masks/
  - flame_param/
  - canonical_flame_param.npz
  - transforms_train.json
  - transforms_val.json
  - transforms_test.json

The source BirdZoom structure is expected to be:
  /rpool/data/P5data/BirdZoom/
    static/
      camera/
      outfit/
      outraw/
      alphas/
    dynamic/
      outraw1/ alphas1/ outfit1/
      outraw2/ alphas2/ outfit2/
      outraw3/ alphas3/ outfit3/

The exporter pads FLAME expression coefficients to 100 and truncates shape
coefficients to 300 so the result matches the current lumio_base FLAME config.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image


DEFAULT_SOURCE_ROOT = Path("/rpool/data/P5data/BirdZoom")
DEFAULT_TARGET_ROOT = Path("/rpool/data/icy/dataset")


@dataclass(frozen=True)
class CameraRecord:
    camera_index: int
    camera_id: str
    focal_x: float
    focal_y: float
    principal_x: float
    principal_y: float
    matrix: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class FlameRecord:
    shape: np.ndarray
    expr: np.ndarray
    rotation: np.ndarray
    neck_pose: np.ndarray
    jaw_pose: np.ndarray
    eyes_pose: np.ndarray
    translation: np.ndarray
    static_offset: np.ndarray
    dynamic_offset: np.ndarray


def parse_camera_file(path: Path, camera_index: int) -> CameraRecord:
    lines = path.read_text().splitlines()

    focal_line = lines[lines.index("#focal length") + 1].split()
    principal_line = lines[lines.index("#principal point") + 1].split()
    matrix_start = lines.index("MATRIX :") + 1
    matrix = np.array([[float(value) for value in line.split()] for line in lines[matrix_start:matrix_start + 4]], dtype=np.float64)

    if matrix.shape != (4, 4):
        raise ValueError(f"Unexpected matrix shape in {path}: {matrix.shape}")

    if np.linalg.det(matrix[:3, :3]) < 0:
        matrix = matrix.copy()
        matrix[:3, 0] *= -1.0

    camera_id = path.stem
    return CameraRecord(
        camera_index=camera_index,
        camera_id=camera_id,
        focal_x=float(focal_line[0]),
        focal_y=float(focal_line[1]),
        principal_x=float(principal_line[0]),
        principal_y=float(principal_line[1]),
        matrix=matrix,
        width=4896,
        height=3684,
    )


def load_flame_text(flame_path: Path, offsets_path: Path) -> FlameRecord:
    flame_lines = flame_path.read_text().strip().splitlines()
    if len(flame_lines) < 3:
        raise ValueError(f"Unexpected FLAME text format in {flame_path}")

    shape = np.fromstring(flame_lines[0], sep=" ", dtype=np.float64)
    expr = np.fromstring(flame_lines[1], sep=" ", dtype=np.float64)
    translation = np.fromstring(flame_lines[2], sep=" ", dtype=np.float64)

    offsets = np.loadtxt(offsets_path, dtype=np.float64)
    if offsets.ndim != 2 or offsets.shape[1] != 3:
        raise ValueError(f"Unexpected offsets shape in {offsets_path}: {offsets.shape}")

    return FlameRecord(
        shape=shape,
        expr=expr,
        rotation=np.zeros(3, dtype=np.float64),
        neck_pose=np.zeros(3, dtype=np.float64),
        jaw_pose=np.zeros(3, dtype=np.float64),
        eyes_pose=np.zeros(6, dtype=np.float64),
        translation=translation,
        static_offset=offsets[None, ...],
        dynamic_offset=np.zeros((1, offsets.shape[0], 3), dtype=np.float64),
    )


def pad_or_truncate(array: np.ndarray, target_length: int) -> np.ndarray:
    flat = np.asarray(array, dtype=np.float64).reshape(-1)
    if flat.shape[0] >= target_length:
        return flat[:target_length]
    result = np.zeros(target_length, dtype=np.float64)
    result[: flat.shape[0]] = flat
    return result


def pad_expr(expr: np.ndarray, target_length: int = 100) -> np.ndarray:
    flat = np.asarray(expr, dtype=np.float64).reshape(-1)
    if flat.shape[0] >= target_length:
        return flat[:target_length][None, ...]
    result = np.zeros((1, target_length), dtype=np.float64)
    result[0, : flat.shape[0]] = flat
    return result


def flame_record_to_npz(record: FlameRecord, *, zero_dynamic: bool = False) -> Dict[str, np.ndarray]:
    dynamic_offset = record.dynamic_offset
    if zero_dynamic:
        dynamic_offset = np.zeros_like(dynamic_offset)

    return {
        "translation": record.translation.reshape(1, 3),
        "rotation": record.rotation.reshape(1, 3),
        "neck_pose": record.neck_pose.reshape(1, 3),
        "jaw_pose": record.jaw_pose.reshape(1, 3),
        "eyes_pose": record.eyes_pose.reshape(1, 6),
        "shape": pad_or_truncate(record.shape, 300),
        "expr": pad_expr(record.expr, 100),
        "static_offset": record.static_offset,
        "dynamic_offset": dynamic_offset,
    }


def load_camera_records(camera_dir: Path) -> List[CameraRecord]:
    camera_files = sorted(camera_dir.glob("camera*.txt"), key=lambda path: int(path.stem.replace("camera", "")))
    records: List[CameraRecord] = []
    for fallback_index, camera_file in enumerate(camera_files):
        camera_number = int(camera_file.stem.replace("camera", ""))
        records.append(parse_camera_file(camera_file, camera_number))
    return records


def resize_image(image: Image.Image, scale: float, *, resample: int) -> Image.Image:
    if scale == 1.0:
        return image
    width = max(1, int(round(image.width * scale)))
    height = max(1, int(round(image.height * scale)))
    return image.resize((width, height), resample=resample)


def save_rgb_and_mask(
    rgb_path: Path,
    mask_path: Path,
    out_rgb_path: Path,
    out_mask_path: Path,
    scale: float,
):
    rgb_image = Image.open(rgb_path).convert("RGB")
    mask_image = Image.open(mask_path).convert("L")

    if scale != 1.0:
        rgb_image = resize_image(rgb_image, scale, resample=Image.Resampling.LANCZOS)
        mask_image = resize_image(mask_image, scale, resample=Image.Resampling.NEAREST)

    out_rgb_path.parent.mkdir(parents=True, exist_ok=True)
    out_mask_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_image.save(out_rgb_path)
    mask_image.convert("RGB").save(out_mask_path)


def build_frame_record(
    *,
    timestep_index: int,
    camera: CameraRecord,
    image_path: str,
    mask_path: str,
    flame_param_path: str,
    scale: float,
) -> Dict[str, object]:
    scaled_width = int(round(camera.width * scale)) if scale != 1.0 else camera.width
    scaled_height = int(round(camera.height * scale)) if scale != 1.0 else camera.height
    fl_x = camera.focal_x * scale
    fl_y = camera.focal_y * scale
    cx = camera.principal_x * scale
    cy = camera.principal_y * scale
    camera_angle_x = 2.0 * math.atan(scaled_width / (2.0 * fl_x))
    camera_angle_y = 2.0 * math.atan(scaled_height / (2.0 * fl_y))

    return {
        "timestep_index": timestep_index,
        "timestep_index_original": timestep_index,
        "timestep_id": f"frame_{timestep_index:05d}",
        "camera_index": camera.camera_index,
        "camera_id": camera.camera_id,
        "file_path": image_path,
        "fg_mask_path": mask_path,
        "flame_param_path": flame_param_path,
        "transform_matrix": camera.matrix.tolist(),
        "cx": cx,
        "cy": cy,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "h": scaled_height,
        "w": scaled_width,
        "camera_angle_x": camera_angle_x,
        "camera_angle_y": camera_angle_y,
    }


def write_transforms(dataset_dir: Path, frames: List[Dict[str, object]]) -> None:
    if not frames:
        raise ValueError(f"No frames to write for {dataset_dir}")

    first = frames[0]
    payload = {
        "timestep_indices": sorted({int(frame["timestep_index"]) for frame in frames}),
        "camera_indices": sorted({int(frame["camera_index"]) for frame in frames}),
        "frames": frames,
        "cx": first["cx"],
        "cy": first["cy"],
        "fl_x": first["fl_x"],
        "fl_y": first["fl_y"],
        "h": first["h"],
        "w": first["w"],
        "camera_angle_x": first["camera_angle_x"],
        "camera_angle_y": first["camera_angle_y"],
    }

    for filename in ["transforms_train.json", "transforms_val.json", "transforms_test.json", "transforms.json"]:
        with (dataset_dir / filename).open("w") as handle:
            json.dump(payload, handle, indent=2)


def convert_sequence(
    source_root: Path,
    target_root: Path,
    sequence_id: int,
    *,
    scale: float,
    overwrite: bool,
) -> Path:
    sequence_name = f"birdzoom_{sequence_id:02d}"
    dataset_dir = target_root / sequence_name

    if dataset_dir.exists() and overwrite:
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    static_dir = source_root / "static"
    dynamic_dir = source_root / "dynamic"
    camera_records = load_camera_records(static_dir / "camera")

    canonical_flame = load_flame_text(static_dir / "outfit" / "Flame_Params.txt", static_dir / "outfit" / "offsets.txt")
    np.savez(dataset_dir / "canonical_flame_param.npz", **flame_record_to_npz(canonical_flame, zero_dynamic=True))

    rgb_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "fg_masks"
    flame_param_dir = dataset_dir / "flame_param"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    flame_param_dir.mkdir(parents=True, exist_ok=True)

    sequence_rgb_dir = dynamic_dir / f"outraw{sequence_id}"
    sequence_mask_dir = dynamic_dir / f"alphas{sequence_id}"
    sequence_flame_dir = dynamic_dir / f"outfit{sequence_id}"

    frames: List[Dict[str, object]] = []
    timestep_dirs = sorted([path for path in sequence_flame_dir.iterdir() if path.is_dir()], key=lambda path: int(path.name))

    for timestep_dir in timestep_dirs:
        timestep_index = int(timestep_dir.name)
        frame_flame = load_flame_text(timestep_dir / "Flame_Params.txt", timestep_dir / "offsets.txt")
        flame_param_path = flame_param_dir / f"{timestep_index:05d}.npz"
        np.savez(flame_param_path, **flame_record_to_npz(frame_flame))

        for camera in camera_records:
            rgb_src = sequence_rgb_dir / f"{timestep_index}_{camera.camera_index}.png"
            mask_src = sequence_mask_dir / f"{timestep_index}_{camera.camera_index}_mask.png"
            if not rgb_src.exists():
                raise FileNotFoundError(rgb_src)
            if not mask_src.exists():
                raise FileNotFoundError(mask_src)

            rgb_rel = f"images/{timestep_index:05d}_{camera.camera_index:02d}.png"
            mask_rel = f"fg_masks/{timestep_index:05d}_{camera.camera_index:02d}.png"
            flame_rel = f"flame_param/{timestep_index:05d}.npz"

            save_rgb_and_mask(
                rgb_src,
                mask_src,
                dataset_dir / rgb_rel,
                dataset_dir / mask_rel,
                scale,
            )

            frames.append(
                build_frame_record(
                    timestep_index=timestep_index,
                    camera=camera,
                    image_path=rgb_rel,
                    mask_path=mask_rel,
                    flame_param_path=flame_rel,
                    scale=scale,
                )
            )

    write_transforms(dataset_dir, frames)
    return dataset_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert BirdZoom captures to VHAP dataset folders.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT, help="BirdZoom root directory")
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT, help="Output dataset root")
    parser.add_argument("--sequence", type=int, nargs="*", default=[1, 2, 3], help="Dynamic sequence IDs to export")
    parser.add_argument("--scale", type=float, default=0.25, help="Resize factor applied to RGB and masks")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directories")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.source_root.exists():
        raise FileNotFoundError(args.source_root)

    exported = []
    for sequence_id in args.sequence:
        dataset_dir = convert_sequence(
            args.source_root,
            args.target_root,
            sequence_id,
            scale=args.scale,
            overwrite=args.overwrite,
        )
        exported.append(dataset_dir)
        print(f"[OK] Exported BirdZoom sequence {sequence_id} to {dataset_dir}")

    print("[DONE] Exported:")
    for path in exported:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())