from __future__ import annotations

"""One-command DuoNeRF demo or real two-image run.

This is the user-facing entry point.  It prepares all metadata/depth assets
that SinNeRF normally expects, then calls the modern end-to-end trainer already
included in this repository.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="앞/뒤 이미지 두 장에서 DuoNeRF 학습과 회전 렌더링까지 실행합니다."
    )
    parser.add_argument("--front", help="정면 이미지 경로")
    parser.add_argument("--back", help="후면 이미지 경로")
    parser.add_argument("--generate_demo", action="store_true")
    parser.add_argument("--work_dir", default="outputs/duonerf_from_images")
    parser.add_argument("--img_wh", nargs=2, type=int, default=[64, 64])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--rays_per_step", type=int, default=64)
    parser.add_argument("--render_frames", type=int, default=12)
    parser.add_argument(
        "--pose_mode",
        choices=["front_back", "auto", "sfm"],
        default="front_back",
    )
    return parser


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parent
    work_dir = Path(args.work_dir).resolve()
    scene_dir = work_dir / "scene"
    result_dir = work_dir / "result"

    prepare_command = [
        sys.executable,
        str(root / "prepare_duonerf_scene.py"),
        "--output_dir",
        str(scene_dir),
        "--img_wh",
        str(args.img_wh[0]),
        str(args.img_wh[1]),
        "--pose_mode",
        args.pose_mode,
        "--overwrite",
    ]
    if args.front and args.back:
        prepare_command.extend(["--front", args.front, "--back", args.back])
    else:
        prepare_command.append("--generate_demo")
    run(prepare_command)

    trainer = root / "run_duonerf_end_to_end.py"
    if not trainer.exists():
        raise FileNotFoundError(
            "run_duonerf_end_to_end.py가 없습니다. 최신 main 브랜치를 사용해 주세요."
        )
    run(
        [
            sys.executable,
            str(trainer),
            "--root_dir",
            str(scene_dir),
            "--ref_indices",
            "0",
            "1",
            "--img_wh",
            str(args.img_wh[0]),
            str(args.img_wh[1]),
            "--steps",
            str(args.steps),
            "--rays_per_step",
            str(args.rays_per_step),
            "--render_frames",
            str(args.render_frames),
            "--output_dir",
            str(result_dir),
        ]
    )

    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    combined = {
        "status": "DUONERF_FROM_IMAGES_OK",
        "scene_dir": str(scene_dir),
        "result_dir": str(result_dir),
        "checkpoint": summary["checkpoint"],
        "gif": summary["gif"],
        "loss_first": summary["loss_first"],
        "loss_last": summary["loss_last"],
    }
    combined_path = work_dir / "run_summary.json"
    combined_path.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("DUONERF_FROM_IMAGES_OK")
    print(json.dumps(combined, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
