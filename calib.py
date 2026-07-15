from __future__ import annotations

"""Checkerboard camera calibration utility used by the original DuoNeRF idea.

Calibration is optional for the default two-image demo because
``prepare_duonerf_scene.py`` can synthesize intrinsics from a field of view.
Use this script when real checkerboard photographs are available.
"""

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="체커보드 이미지로 카메라 내부 파라미터를 계산합니다.")
    parser.add_argument("--images", required=True, help="예: imgs_calib/*.jpg")
    parser.add_argument("--cols", type=int, default=10, help="체커보드 내부 코너 열 개수")
    parser.add_argument("--rows", type=int, default=7, help="체커보드 내부 코너 행 개수")
    parser.add_argument("--square_size", type=float, default=22.0, help="한 칸 실제 길이")
    parser.add_argument("--output_dir", default="outputs/calibration")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_paths = [Path(path) for path in sorted(glob.glob(args.images))]
    if not image_paths:
        raise FileNotFoundError(f"체커보드 이미지를 찾지 못했습니다: {args.images}")

    object_template = np.zeros((args.rows * args.cols, 3), np.float32)
    object_template[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    object_template *= args.square_size

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    used: list[str] = []
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-3,
    )

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(
            gray,
            (args.cols, args.rows),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            continue
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(object_template.copy())
        image_points.append(refined)
        used.append(str(path))

    if image_size is None or len(object_points) < 3:
        raise RuntimeError(
            f"보정에 사용할 수 있는 체커보드 검출이 부족합니다: {len(object_points)}장"
        )

    rms, intrinsic, distortion, rotations, translations = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    total_error = 0.0
    for index, points in enumerate(object_points):
        projected, _ = cv2.projectPoints(
            points,
            rotations[index],
            translations[index],
            intrinsic,
            distortion,
        )
        total_error += cv2.norm(image_points[index], projected, cv2.NORM_L2) / len(projected)
    mean_error = total_error / len(object_points)

    output_dir = Path(args.output_dir)
    undistorted_dir = output_dir / "undistorted"
    undistorted_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "camera_calibration.npz",
        K=intrinsic,
        dist=distortion,
        image_size=np.asarray(image_size),
    )
    summary = {
        "status": "DUONERF_CALIBRATION_OK",
        "rms": float(rms),
        "mean_reprojection_error": float(mean_error),
        "image_size": list(image_size),
        "used_images": used,
        "K": intrinsic.tolist(),
        "dist": distortion.reshape(-1).tolist(),
    }
    (output_dir / "camera_calibration.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        undistorted = cv2.undistort(image, intrinsic, distortion)
        cv2.imwrite(str(undistorted_dir / path.name), undistorted)

    print("DUONERF_CALIBRATION_OK")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
