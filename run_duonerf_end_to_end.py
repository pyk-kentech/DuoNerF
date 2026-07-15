from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from datasets.blender_ray_patch_2image_rot3d import (
    Blender_ray_patch_2image_rot3d_Dataset,
)
from models.nerf import Embedding, NeRF
from models.rendering import render_rays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", required=True)
    parser.add_argument(
        "--ref_indices",
        nargs=2,
        type=int,
        required=True,
    )
    parser.add_argument(
        "--img_wh",
        nargs=2,
        type=int,
        default=[32, 32],
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--rays_per_step",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--render_frames",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/duonerf_cpu_demo",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    return parser.parse_args()


def render_batch(
    models: list[NeRF],
    embeddings: list[Embedding],
    rays: torch.Tensor,
    white_back: bool,
) -> dict[str, torch.Tensor]:
    return render_rays(
        models=models,
        embeddings=embeddings,
        rays=rays,
        N_samples=8,
        use_disp=False,
        perturb=0.0,
        noise_std=0.0,
        N_importance=8,
        chunk=4096,
        white_back=white_back,
    )


def calculate_loss(
    output: dict[str, torch.Tensor],
    rgb_target: torch.Tensor,
    depth_target: torch.Tensor,
) -> torch.Tensor:
    rgb_loss = F.mse_loss(
        output["rgb_coarse"],
        rgb_target,
    )

    rgb_loss = rgb_loss + F.mse_loss(
        output["rgb_fine"],
        rgb_target,
    )

    depth_loss = F.smooth_l1_loss(
        output["depth_coarse"].unsqueeze(-1),
        depth_target,
    )

    depth_loss = depth_loss + F.smooth_l1_loss(
        output["depth_fine"].unsqueeze(-1),
        depth_target,
    )

    return rgb_loss + 0.1 * depth_loss


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.set_num_threads(
        max(
            1,
            min(4, torch.get_num_threads()),
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_dir = Path(args.output_dir)
    frame_dir = output_dir / "frames"

    frame_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_dataset = (
        Blender_ray_patch_2image_rot3d_Dataset(
            root_dir=args.root_dir,
            split="train",
            img_wh=tuple(args.img_wh),
            patch_size=8,
            with_ref=True,
            load_depth=True,
            depth_type="nerf",
            sH=1,
            sW=1,
            angle=5,
            ref_indices=args.ref_indices,
            duo_samples_per_epoch=max(
                2,
                args.steps,
            ),
            duo_random_rays=max(
                32,
                args.rays_per_step,
            ),
            duo_pseudo_views=1,
        )
    )

    embeddings = [
        Embedding(3, 10).to(device),
        Embedding(3, 4).to(device),
    ]

    models = [
        NeRF(
            use_new_activation=True
        ).to(device),
        NeRF(
            use_new_activation=True
        ).to(device),
    ]

    parameters = [
        parameter
        for model in models
        for parameter in model.parameters()
    ]

    optimizer = torch.optim.Adam(
        parameters,
        lr=1e-4,
    )

    rays_per_view = max(
        1,
        args.rays_per_step // 2,
    )

    losses: list[float] = []

    models[0].train()
    models[1].train()

    for step in range(
        1,
        args.steps + 1,
    ):
        ray_parts = []
        rgb_parts = []
        depth_parts = []

        # 매 단계마다 앞면과 뒷면에서 같은 수의 광선을 추출합니다.
        for reference_slot in range(2):
            reference_rays = (
                train_dataset.ref_rays[
                    reference_slot
                ].reshape(-1, 8)
            )

            reference_rgb = (
                train_dataset.ref_views[
                    reference_slot
                ].reshape(-1, 3)
            )

            reference_depth = (
                train_dataset.ref_depths[
                    reference_slot
                ].reshape(-1, 1)
            )

            indices = torch.randint(
                low=0,
                high=reference_rays.shape[0],
                size=(rays_per_view,),
            )

            ray_parts.append(
                reference_rays[indices]
            )
            rgb_parts.append(
                reference_rgb[indices]
            )
            depth_parts.append(
                reference_depth[indices]
            )

        rays = torch.cat(
            ray_parts,
            dim=0,
        ).to(device)

        rgb_target = torch.cat(
            rgb_parts,
            dim=0,
        ).to(device)

        depth_target = torch.cat(
            depth_parts,
            dim=0,
        ).to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        output = render_batch(
            models=models,
            embeddings=embeddings,
            rays=rays,
            white_back=train_dataset.white_back,
        )

        loss = calculate_loss(
            output=output,
            rgb_target=rgb_target,
            depth_target=depth_target,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "비정상 손실값: "
                f"step={step}, "
                f"loss={loss.item()}"
            )

        loss.backward()
        optimizer.step()

        loss_value = float(
            loss.detach().cpu()
        )

        losses.append(loss_value)

        report_interval = max(
            1,
            args.steps // 5,
        )

        if (
            step == 1
            or step == args.steps
            or step % report_interval == 0
        ):
            print(
                f"step={step}/{args.steps} "
                f"loss={loss_value:.6f}"
            )

    checkpoint_path = (
        output_dir
        / "duonerf_cpu_demo.pt"
    )

    torch.save(
        {
            "nerf_coarse": (
                models[0].state_dict()
            ),
            "nerf_fine": (
                models[1].state_dict()
            ),
            "optimizer": (
                optimizer.state_dict()
            ),
            "ref_indices": list(
                args.ref_indices
            ),
            "img_wh": list(
                args.img_wh
            ),
            "steps": args.steps,
            "losses": losses,
        },
        checkpoint_path,
    )

    eval_dataset = (
        Blender_ray_patch_2image_rot3d_Dataset(
            root_dir=args.root_dir,
            split="test_train",
            img_wh=tuple(args.img_wh),
            patch_size=8,
            ref_indices=args.ref_indices,
        )
    )

    models[0].eval()
    models[1].eval()

    image_width = args.img_wh[0]
    image_height = args.img_wh[1]

    frame_count = max(
        1,
        min(
            args.render_frames,
            len(eval_dataset),
        ),
    )

    frame_indices = np.linspace(
        0,
        len(eval_dataset) - 1,
        frame_count,
        dtype=int,
    )

    rendered_frames = []

    with torch.no_grad():
        for output_index, dataset_index in enumerate(
            frame_indices
        ):
            rays = eval_dataset[
                int(dataset_index)
            ]["rays"]

            rgb_chunks = []

            for start in range(
                0,
                rays.shape[0],
                128,
            ):
                ray_chunk = rays[
                    start:start + 128
                ].to(device)

                rendered = render_batch(
                    models=models,
                    embeddings=embeddings,
                    rays=ray_chunk,
                    white_back=(
                        eval_dataset.white_back
                    ),
                )

                rgb_chunks.append(
                    rendered[
                        "rgb_fine"
                    ].cpu()
                )

            image = torch.cat(
                rgb_chunks,
                dim=0,
            ).reshape(
                image_height,
                image_width,
                3,
            )

            image = torch.nan_to_num(
                image
            ).clamp(
                0.0,
                1.0,
            )

            image_uint8 = (
                image.numpy()
                * 255.0
            ).astype(
                np.uint8
            )

            frame_path = (
                frame_dir
                / f"{output_index:03d}.png"
            )

            imageio.imwrite(
                frame_path,
                image_uint8,
            )

            rendered_frames.append(
                image_uint8
            )

            print(
                "rendered_frame="
                f"{output_index + 1}/"
                f"{frame_count}"
            )

    gif_path = (
        output_dir
        / "duonerf_orbit.gif"
    )

    imageio.mimsave(
        gif_path,
        rendered_frames,
        duration=0.15,
        loop=0,
    )

    summary = {
        "status": (
            "DUONERF_END_TO_END_OK"
        ),
        "device": str(device),
        "root_dir": str(
            Path(
                args.root_dir
            ).resolve()
        ),
        "ref_indices": list(
            args.ref_indices
        ),
        "steps": args.steps,
        "rays_per_step_actual": (
            rays_per_view * 2
        ),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "checkpoint": str(
            checkpoint_path.resolve()
        ),
        "gif": str(
            gif_path.resolve()
        ),
        "frame_count": frame_count,
    }

    summary_path = (
        output_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    if (
        not checkpoint_path.exists()
        or checkpoint_path.stat().st_size == 0
    ):
        raise RuntimeError(
            "체크포인트 생성 실패"
        )

    if (
        not gif_path.exists()
        or gif_path.stat().st_size == 0
    ):
        raise RuntimeError(
            "회전 GIF 생성 실패"
        )

    print(
        "DUONERF_END_TO_END_OK"
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
