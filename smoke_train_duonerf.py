from __future__ import annotations

import argparse
import math

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
        "--rays_per_view",
        type=int,
        default=32,
    )
    return parser.parse_args()


def render(
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


def compute_loss(
    output: dict[str, torch.Tensor],
    rgb_target: torch.Tensor,
    depth_target: torch.Tensor,
) -> torch.Tensor:
    loss = F.mse_loss(
        output["rgb_coarse"],
        rgb_target,
    )

    loss = loss + 0.1 * F.smooth_l1_loss(
        output["depth_coarse"].unsqueeze(-1),
        depth_target,
    )

    if "rgb_fine" in output:
        loss = loss + F.mse_loss(
            output["rgb_fine"],
            rgb_target,
        )

        loss = loss + 0.1 * F.smooth_l1_loss(
            output["depth_fine"].unsqueeze(-1),
            depth_target,
        )

    return loss


def main() -> None:
    args = parse_args()

    torch.manual_seed(2026)
    torch.set_num_threads(
        max(1, min(4, torch.get_num_threads()))
    )

    dataset = Blender_ray_patch_2image_rot3d_Dataset(
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
        duo_samples_per_epoch=2,
        duo_random_rays=64,
        duo_pseudo_views=1,
    )

    rays_per_view = max(1, int(args.rays_per_view))

    # 앞면과 뒷면에서 동일한 수의 광선을 가져옵니다.
    rays = torch.cat(
        [
            dataset.ref_rays[0]
            .reshape(-1, 8)[:rays_per_view],
            dataset.ref_rays[1]
            .reshape(-1, 8)[:rays_per_view],
        ],
        dim=0,
    )

    rgb_targets = torch.cat(
        [
            dataset.ref_views[0]
            .reshape(-1, 3)[:rays_per_view],
            dataset.ref_views[1]
            .reshape(-1, 3)[:rays_per_view],
        ],
        dim=0,
    )

    depth_targets = torch.cat(
        [
            dataset.ref_depths[0]
            .reshape(-1, 1)[:rays_per_view],
            dataset.ref_depths[1]
            .reshape(-1, 1)[:rays_per_view],
        ],
        dim=0,
    )

    embeddings = [
        Embedding(3, 10),
        Embedding(3, 4),
    ]

    models = [
        NeRF(use_new_activation=True),
        NeRF(use_new_activation=True),
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

    optimizer.zero_grad(set_to_none=True)

    output_before = render(
        models,
        embeddings,
        rays,
        dataset.white_back,
    )

    loss_before = compute_loss(
        output_before,
        rgb_targets,
        depth_targets,
    )

    if not torch.isfinite(loss_before):
        raise RuntimeError(
            f"비정상 손실값: {loss_before.item()}"
        )

    loss_before.backward()

    gradient_square_sum = 0.0

    for parameter in parameters:
        if parameter.grad is not None:
            gradient_square_sum += float(
                parameter.grad.detach().pow(2).sum()
            )

    gradient_norm = math.sqrt(
        gradient_square_sum
    )

    if (
        not math.isfinite(gradient_norm)
        or gradient_norm <= 0.0
    ):
        raise RuntimeError(
            f"비정상 gradient norm: {gradient_norm}"
        )

    optimizer.step()

    with torch.no_grad():
        output_after = render(
            models,
            embeddings,
            rays,
            dataset.white_back,
        )

        loss_after = compute_loss(
            output_after,
            rgb_targets,
            depth_targets,
        )

    print("DUONERF_TRAIN_STEP_OK")
    print(
        f"reference_indices={dataset.ref_indices}"
    )
    print(
        f"rays_total={rays.shape[0]}"
    )
    print(
        f"loss_before={loss_before.item():.6f}"
    )
    print(
        f"loss_after={loss_after.item():.6f}"
    )
    print(
        f"gradient_norm={gradient_norm:.6f}"
    )
    print(
        "coarse_rgb_shape="
        f"{tuple(output_after['rgb_coarse'].shape)}"
    )
    print(
        "fine_rgb_shape="
        f"{tuple(output_after['rgb_fine'].shape)}"
    )


if __name__ == "__main__":
    main()
