from __future__ import annotations

"""Two-view geometry diagnostic for DuoNeRF front/back inputs."""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from prepare_duonerf_scene import _attempt_relative_pose


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="두 이미지의 특징 매칭과 상대 카메라 자세를 계산합니다.")
    parser.add_argument("--front", required=True)
    parser.add_argument("--back", required=True)
    parser.add_argument("--fov_degrees", type=float, default=50.0)
    parser.add_argument("--output_dir", default="outputs/sfm")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    front = np.asarray(Image.open(args.front).convert("RGB"), dtype=np.uint8)
    back_image = Image.open(args.back).convert("RGB")
    if back_image.size != (front.shape[1], front.shape[0]):
        back_image = back_image.resize((front.shape[1], front.shape[0]), Image.Resampling.LANCZOS)
    back = np.asarray(back_image, dtype=np.uint8)

    height, width = front.shape[:2]
    focal = 0.5 * width / math.tan(0.5 * math.radians(args.fov_degrees))
    intrinsic = np.array(
        [[focal, 0.0, (width - 1.0) / 2.0],
         [0.0, focal, (height - 1.0) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    relative_pose, diagnostics, matches = _attempt_relative_pose(front, back, intrinsic)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if matches is not None:
        cv2.imwrite(str(output_dir / "matches.png"), matches)
    if relative_pose is not None:
        np.save(output_dir / "relative_c2w.npy", relative_pose)

    result = {
        "status": "DUONERF_SFM_OK" if relative_pose is not None else "DUONERF_SFM_FALLBACK_REQUIRED",
        "K": intrinsic.tolist(),
        "relative_c2w": None if relative_pose is None else relative_pose.tolist(),
        "diagnostics": asdict(diagnostics),
    }
    (output_dir / "sfm_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(result["status"])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
