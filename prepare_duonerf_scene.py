from __future__ import annotations

"""Prepare a runnable DuoNeRF scene from only two ordinary images.

The original SinNeRF data loader expects Blender-style camera transforms and a
per-reference depth map.  This script creates those missing assets so a user
only needs a front image and a back image.  When no images are supplied it also
creates a deterministic demo pair, which makes the repository runnable out of
the box.

The generated geometry is intentionally a prior, not ground truth:

* camera poses default to an opposing front/back pair;
* depth is a smooth silhouette-aware proxy built from each image mask;
* optional two-view SfM diagnostics are attempted when the images overlap.

That is enough to exercise the complete two-reference NeRF pipeline even when
results are not the main goal.
"""

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass
class PoseDiagnostics:
    method: str
    feature_type: str
    keypoints_front: int
    keypoints_back: int
    raw_matches: int
    good_matches: int
    essential_inliers: int
    recovered_pose: bool
    fallback_used: bool
    message: str


def _rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c = math.cos(angle)
    s = math.sin(angle)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )
    return matrix


def _front_back_poses(radius: float) -> tuple[np.ndarray, np.ndarray]:
    front = np.eye(4, dtype=np.float64)
    front[2, 3] = radius

    back = _rotation_y(180.0)
    back[2, 3] = -radius
    return front, back


def _look_at_pose(position: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """Return a NeRF/Blender-style camera-to-world pose.

    Local camera forward is -Z, up is +Y and right is +X.
    """

    if target is None:
        target = np.zeros(3, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    backward = position - target
    backward /= max(np.linalg.norm(backward), 1e-8)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, backward)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(world_up, backward)
    right /= max(np.linalg.norm(right), 1e-8)
    up = np.cross(backward, right)
    up /= max(np.linalg.norm(up), 1e-8)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = backward
    pose[:3, 3] = position
    return pose


def _create_demo_pair(root: Path, width: int, height: int) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    front_path = root / "front.png"
    back_path = root / "back.png"

    def make_image(side: str) -> Image.Image:
        image = Image.new("RGBA", (width, height), (247, 247, 247, 255))
        draw = ImageDraw.Draw(image)
        margin = max(4, int(min(width, height) * 0.12))
        left, top = margin, margin
        right, bottom = width - margin, height - margin

        # Shared silhouette so the two views describe one object.
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=max(3, int(min(width, height) * 0.12)),
            fill=(50, 90, 150, 255) if side == "front" else (125, 70, 150, 255),
            outline=(25, 25, 35, 255),
            width=max(1, width // 64),
        )
        cx = (left + right) // 2
        cy = (top + bottom) // 2
        r = max(3, int(min(width, height) * 0.12))
        if side == "front":
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(235, 170, 55, 255))
            draw.polygon(
                [(cx, top + margin // 2), (cx - r, top + 2 * r), (cx + r, top + 2 * r)],
                fill=(235, 235, 245, 255),
            )
            draw.text((left + 4, bottom - 16), "FRONT", fill=(255, 255, 255, 255))
        else:
            draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=(70, 190, 160, 255))
            draw.line((left + r, top + r, right - r, bottom - r), fill=(240, 230, 100, 255), width=max(2, width // 32))
            draw.text((left + 4, bottom - 16), "BACK", fill=(255, 255, 255, 255))

        # A slight shadow makes the foreground detector exercise a realistic case.
        shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.ellipse(
            (left + width // 20, bottom - height // 20, right + width // 20, bottom + height // 18),
            fill=(0, 0, 0, 55),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(1, width // 80)))
        return Image.alpha_composite(shadow, image)

    make_image("front").save(front_path)
    make_image("back").save(back_path)
    return front_path, back_path


def _letterbox_rgba(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    image = image.convert("RGBA")
    scale = min(target_w / image.width, target_h / image.height)
    new_w = max(1, int(round(image.width * scale)))
    new_h = max(1, int(round(image.height * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 255))
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    canvas.alpha_composite(resized, dest=offset)
    return canvas


def _largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary * 255
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas)) + 1
    return (labels == label).astype(np.uint8) * 255


def _estimate_foreground_mask(rgba: np.ndarray, mode: str) -> np.ndarray:
    height, width = rgba.shape[:2]
    if mode == "full":
        return np.full((height, width), 255, dtype=np.uint8)

    alpha = rgba[..., 3]
    if mode in {"alpha", "auto"} and np.any(alpha < 250) and np.any(alpha > 5):
        mask = (alpha > 16).astype(np.uint8) * 255
    else:
        rgb = rgba[..., :3].astype(np.float32)
        border_width = max(1, min(height, width) // 20)
        border = np.concatenate(
            [
                rgb[:border_width].reshape(-1, 3),
                rgb[-border_width:].reshape(-1, 3),
                rgb[:, :border_width].reshape(-1, 3),
                rgb[:, -border_width:].reshape(-1, 3),
            ],
            axis=0,
        )
        background = np.median(border, axis=0)
        distance = np.linalg.norm(rgb - background[None, None, :], axis=2)
        border_distance = np.linalg.norm(border - background[None, :], axis=1)
        adaptive = float(np.percentile(border_distance, 95)) + 12.0
        threshold = max(16.0, min(72.0, adaptive))
        mask = (distance > threshold).astype(np.uint8) * 255

    kernel_size = max(3, (min(height, width) // 32) | 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = _largest_component(mask)

    foreground_ratio = float(np.mean(mask > 0))
    if foreground_ratio < 0.02 or foreground_ratio > 0.98:
        # A bad background estimate is less useful than treating the full frame as valid.
        mask = np.full((height, width), 255, dtype=np.uint8)
    return mask


def _build_depth_prior(mask: np.ndarray, radius: float) -> np.ndarray:
    foreground = (mask > 0).astype(np.uint8)
    distance = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
    if float(distance.max()) > 0:
        distance = distance / float(distance.max())

    # The object centre is slightly closer than the silhouette.  Background is
    # placed behind the object so rays remain finite and the training loader can
    # use every pixel without special cases.
    depth = np.full(mask.shape, radius + 0.75, dtype=np.float32)
    object_depth = radius - 0.25 - 0.65 * distance.astype(np.float32)
    depth[foreground > 0] = object_depth[foreground > 0]
    return depth


def _feature_detector() -> tuple[Any, str, int]:
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=4000), "SIFT", cv2.NORM_L2
    return cv2.ORB_create(nfeatures=4000), "ORB", cv2.NORM_HAMMING


def _attempt_relative_pose(
    front_rgb: np.ndarray,
    back_rgb: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray | None, PoseDiagnostics, np.ndarray | None]:
    detector, feature_name, norm_type = _feature_detector()
    front_gray = cv2.cvtColor(front_rgb, cv2.COLOR_RGB2GRAY)
    back_gray = cv2.cvtColor(back_rgb, cv2.COLOR_RGB2GRAY)
    keypoints_a, descriptors_a = detector.detectAndCompute(front_gray, None)
    keypoints_b, descriptors_b = detector.detectAndCompute(back_gray, None)

    kp_a_count = 0 if keypoints_a is None else len(keypoints_a)
    kp_b_count = 0 if keypoints_b is None else len(keypoints_b)
    if descriptors_a is None or descriptors_b is None or kp_a_count < 8 or kp_b_count < 8:
        diagnostic = PoseDiagnostics(
            method="sfm",
            feature_type=feature_name,
            keypoints_front=kp_a_count,
            keypoints_back=kp_b_count,
            raw_matches=0,
            good_matches=0,
            essential_inliers=0,
            recovered_pose=False,
            fallback_used=True,
            message="특징점이 부족해 정면/후면 기본 자세를 사용했습니다.",
        )
        return None, diagnostic, None

    matcher = cv2.BFMatcher(norm_type)
    pairs = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good = []
    for pair in pairs:
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            good.append(pair[0])

    match_visual = cv2.drawMatches(
        cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR),
        keypoints_a,
        cv2.cvtColor(back_rgb, cv2.COLOR_RGB2BGR),
        keypoints_b,
        good[:120],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    if len(good) < 8:
        diagnostic = PoseDiagnostics(
            method="sfm",
            feature_type=feature_name,
            keypoints_front=kp_a_count,
            keypoints_back=kp_b_count,
            raw_matches=len(pairs),
            good_matches=len(good),
            essential_inliers=0,
            recovered_pose=False,
            fallback_used=True,
            message="유효 매칭이 부족해 정면/후면 기본 자세를 사용했습니다.",
        )
        return None, diagnostic, match_visual

    points_a = np.float64([keypoints_a[m.queryIdx].pt for m in good])
    points_b = np.float64([keypoints_b[m.trainIdx].pt for m in good])
    essential, mask = cv2.findEssentialMat(
        points_a,
        points_b,
        intrinsic,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.5,
    )
    if essential is None:
        diagnostic = PoseDiagnostics(
            method="sfm",
            feature_type=feature_name,
            keypoints_front=kp_a_count,
            keypoints_back=kp_b_count,
            raw_matches=len(pairs),
            good_matches=len(good),
            essential_inliers=0,
            recovered_pose=False,
            fallback_used=True,
            message="Essential Matrix 계산에 실패해 기본 자세를 사용했습니다.",
        )
        return None, diagnostic, match_visual

    _, rotation, translation, pose_mask = cv2.recoverPose(
        essential,
        points_a,
        points_b,
        intrinsic,
        mask=mask,
    )
    inliers = int(np.count_nonzero(pose_mask))
    if inliers < 8:
        diagnostic = PoseDiagnostics(
            method="sfm",
            feature_type=feature_name,
            keypoints_front=kp_a_count,
            keypoints_back=kp_b_count,
            raw_matches=len(pairs),
            good_matches=len(good),
            essential_inliers=inliers,
            recovered_pose=False,
            fallback_used=True,
            message="자세 복원 인라이어가 부족해 기본 자세를 사용했습니다.",
        )
        return None, diagnostic, match_visual

    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = translation[:, 0]
    relative_c2w = np.linalg.inv(world_to_camera)
    diagnostic = PoseDiagnostics(
        method="sfm",
        feature_type=feature_name,
        keypoints_front=kp_a_count,
        keypoints_back=kp_b_count,
        raw_matches=len(pairs),
        good_matches=len(good),
        essential_inliers=inliers,
        recovered_pose=True,
        fallback_used=False,
        message="두 영상의 겹치는 특징으로 상대 자세를 복원했습니다.",
    )
    return relative_c2w, diagnostic, match_visual


def _save_prepared_image(
    source: Path,
    destination: Path,
    size: tuple[int, int],
    mask_mode: str,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = _letterbox_rgba(Image.open(source), size)
    rgba = np.asarray(image, dtype=np.uint8)
    mask = _estimate_foreground_mask(rgba, mask_mode)

    # Preserve RGB but replace alpha with the estimated foreground.  SinNeRF's
    # loader composites transparent background onto white.
    prepared = rgba.copy()
    prepared[..., 3] = mask
    Image.fromarray(prepared, mode="RGBA").save(destination)
    depth = _build_depth_prior(mask, radius)
    return prepared[..., :3], mask, depth


def prepare_scene(args: argparse.Namespace) -> dict[str, Any]:
    width, height = args.img_wh
    if width <= 0 or height <= 0:
        raise ValueError("--img_wh는 양의 정수여야 합니다.")
    if width != height:
        raise ValueError("현재 DuoNeRF 데이터 경로는 정사각형 해상도를 사용합니다.")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    train_dir = output_dir / "train"
    depth_dir = output_dir / "depth_nerf"
    diagnostics_dir = output_dir / "diagnostics"
    train_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_demo or (args.front is None and args.back is None):
        front_source, back_source = _create_demo_pair(
            output_dir / "source_images",
            width,
            height,
        )
        source_kind = "generated_demo"
    else:
        if args.front is None or args.back is None:
            raise ValueError("실제 입력을 사용할 때는 --front와 --back을 모두 지정해야 합니다.")
        front_source = Path(args.front).expanduser().resolve()
        back_source = Path(args.back).expanduser().resolve()
        if not front_source.exists() or not back_source.exists():
            raise FileNotFoundError(
                f"입력 이미지를 찾지 못했습니다: front={front_source}, back={back_source}"
            )
        source_kind = "user_images"

    front_rgb, front_mask, front_depth = _save_prepared_image(
        front_source,
        train_dir / "front.png",
        (width, height),
        args.mask_mode,
        args.camera_radius,
    )
    back_rgb, back_mask, back_depth = _save_prepared_image(
        back_source,
        train_dir / "back.png",
        (width, height),
        args.mask_mode,
        args.camera_radius,
    )
    np.save(depth_dir / "front.npy", front_depth.astype(np.float32))
    np.save(depth_dir / "back.npy", back_depth.astype(np.float32))
    Image.fromarray(front_mask).save(diagnostics_dir / "front_mask.png")
    Image.fromarray(back_mask).save(diagnostics_dir / "back_mask.png")

    fov_radians = math.radians(args.fov_degrees)
    focal = 0.5 * width / math.tan(0.5 * fov_radians)
    intrinsic = np.array(
        [[focal, 0.0, (width - 1.0) / 2.0],
         [0.0, focal, (height - 1.0) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    front_pose, back_pose = _front_back_poses(args.camera_radius)
    recovered_relative, diagnostics, match_visual = _attempt_relative_pose(
        front_rgb,
        back_rgb,
        intrinsic,
    )
    if match_visual is not None:
        cv2.imwrite(str(diagnostics_dir / "feature_matches.png"), match_visual)

    if args.pose_mode == "sfm":
        if recovered_relative is None:
            diagnostics.fallback_used = True
        else:
            # Scale the unit translation to a practical camera radius and orient
            # the camera toward the origin.  This converts an SfM direction into
            # a stable NeRF camera-to-world pose.
            direction = recovered_relative[:3, 3]
            if np.linalg.norm(direction) < 1e-8:
                direction = np.array([0.0, 0.0, -1.0])
            direction = direction / np.linalg.norm(direction)
            position = direction * args.camera_radius
            back_pose = _look_at_pose(position)
    elif args.pose_mode == "auto" and recovered_relative is not None and diagnostics.essential_inliers >= args.min_sfm_inliers:
        direction = recovered_relative[:3, 3]
        if np.linalg.norm(direction) >= 1e-8:
            direction = direction / np.linalg.norm(direction)
            position = direction * args.camera_radius
            candidate = _look_at_pose(position)
            # Do not let a near-duplicate pose replace the intended opposite view.
            cosine = float(np.dot(position, np.array([0.0, 0.0, args.camera_radius])) / (args.camera_radius ** 2))
            if cosine < 0.5:
                back_pose = candidate
                diagnostics.fallback_used = False
            else:
                diagnostics.fallback_used = True
                diagnostics.message += " 복원 자세가 정면과 너무 가까워 기본 후면 자세를 유지했습니다."
    else:
        diagnostics.fallback_used = True
        if args.pose_mode == "front_back":
            diagnostics.message = "명시적으로 정면/후면 180도 기본 자세를 사용했습니다."

    transforms = {
        "camera_angle_x": fov_radians,
        "w": width,
        "h": height,
        "fl_x": focal,
        "fl_y": focal,
        "cx": (width - 1.0) / 2.0,
        "cy": (height - 1.0) / 2.0,
        "frames": [
            {
                "file_path": "./train/front",
                "transform_matrix": front_pose.tolist(),
            },
            {
                "file_path": "./train/back",
                "transform_matrix": back_pose.tolist(),
            },
        ],
    }
    (output_dir / "transforms_train.json").write_text(
        json.dumps(transforms, indent=2),
        encoding="utf-8",
    )
    (diagnostics_dir / "pose_diagnostics.json").write_text(
        json.dumps(asdict(diagnostics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = {
        "status": "DUONERF_SCENE_PREPARED",
        "source_kind": source_kind,
        "front_source": str(front_source),
        "back_source": str(back_source),
        "output_dir": str(output_dir),
        "img_wh": [width, height],
        "ref_indices": [0, 1],
        "camera_radius": args.camera_radius,
        "fov_degrees": args.fov_degrees,
        "pose_mode": args.pose_mode,
        "pose_diagnostics": asdict(diagnostics),
        "front_foreground_ratio": float(np.mean(front_mask > 0)),
        "back_foreground_ratio": float(np.mean(back_mask > 0)),
        "files": {
            "transforms": "transforms_train.json",
            "front_image": "train/front.png",
            "back_image": "train/back.png",
            "front_depth": "depth_nerf/front.npy",
            "back_depth": "depth_nerf/back.npy",
        },
    }
    (output_dir / "scene_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="앞/뒤 이미지 두 장만으로 DuoNeRF 학습 장면을 준비합니다."
    )
    parser.add_argument("--front", help="정면 이미지 경로")
    parser.add_argument("--back", help="후면 이미지 경로")
    parser.add_argument(
        "--generate_demo",
        action="store_true",
        help="입력 이미지 없이 실행 가능한 예제 두 장을 생성합니다.",
    )
    parser.add_argument("--output_dir", default="data/duonerf_scene")
    parser.add_argument("--img_wh", nargs=2, type=int, default=[64, 64])
    parser.add_argument("--camera_radius", type=float, default=4.0)
    parser.add_argument("--fov_degrees", type=float, default=50.0)
    parser.add_argument(
        "--pose_mode",
        choices=["front_back", "auto", "sfm"],
        default="front_back",
        help="카메라 자세 방식. front_back은 항상 반대 방향 두 자세를 만듭니다.",
    )
    parser.add_argument(
        "--mask_mode",
        choices=["auto", "alpha", "full"],
        default="auto",
    )
    parser.add_argument("--min_sfm_inliers", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_scene(args)
    print("DUONERF_SCENE_PREPARED")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
