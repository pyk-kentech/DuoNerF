"""Two-reference Blender dataset used by DuoNeRF.

This keeps the SinNeRF training contract, but builds supervision from two
fixed reference views (normally a front/back pair) instead of one image.
The returned dictionary intentionally matches the keys consumed by
``models/sinnerf.py`` so the original two-stage training code remains usable.
"""

import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .ray_utils import get_ray_directions, get_rays
from .blender_ray_patch_1image_rot3d import (
    convert,
    flatten,
    forward_warp,
    rotate,
    rotate_3d,
)


_DEFAULT_REFERENCE_PAIRS = {
    "lego": (20, 70),
    "chair": (99, 49),
    "ship": (80, 30),
    "hotdog": (3, 53),
    "mic": (15, 65),
    "ficus": (22, 72),
    "drums": (19, 69),
    "materials": (20, 70),
}


def _frame_image_path(root_dir, frame):
    rel = frame["file_path"]
    if os.path.splitext(rel)[1]:
        return os.path.join(root_dir, rel)
    return os.path.join(root_dir, rel + ".png")


def _load_rgba_as_rgb(path, img_wh, transform):
    img = Image.open(path).convert("RGBA")
    img = img.resize(tuple(img_wh), Image.LANCZOS)
    img = transform(img)
    return img[:3] * img[3:4] + (1.0 - img[3:4])


def _resize_depth(depth, img_wh):
    width, height = img_wh
    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.shape != (height, width):
        depth = cv2.resize(depth.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(depth.astype(np.float32))


class Blender_ray_patch_2image_rot3d_Dataset(Dataset):
    """SinNeRF-compatible two-reference dataset.

    ``ref_indices`` must contain the two frame indices used as reference
    images. When omitted, a front/back-like pair is selected from the scene
    name. The NeRF itself is shared; both views contribute RGB, depth and
    pseudo-view supervision.
    """

    def __init__(
        self,
        root_dir,
        split="train",
        img_wh=(800, 800),
        patch_size=-1,
        factor=1,
        test_crop=False,
        with_ref=False,
        repeat=1,
        load_depth=False,
        depth_type="nerf",
        sH=1,
        sW=1,
        angle=30,
        ref_indices=None,
        duo_samples_per_epoch=128,
        duo_random_rays=4096,
        duo_pseudo_views=8,
        **kwargs
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_wh = tuple(img_wh)
        self.patch_size = int(patch_size)
        self.factor = factor
        self.test_crop = test_crop
        self.with_ref = with_ref
        self.repeat = max(1, int(repeat))
        self.load_depth = load_depth
        self.depth_type = depth_type
        self.sH = int(sH)
        self.sW = int(sW)
        self.angle = int(angle)
        self.samples_per_epoch = max(2, int(duo_samples_per_epoch))
        self.random_ray_count = max(32, int(duo_random_rays))
        self.pseudo_view_count = max(1, int(duo_pseudo_views))
        self.white_back = True
        self.transform = T.ToTensor()

        if self.img_wh[0] != self.img_wh[1]:
            raise ValueError("DuoNeRF Blender 경로는 현재 정사각형 img_wh를 사용해야 합니다.")
        if self.patch_size <= 0:
            raise ValueError("--patch_size는 1 이상의 값이어야 합니다.")
        if not self.load_depth and self.split == "train":
            raise ValueError("DuoNeRF 학습에는 --load_depth가 필요합니다.")

        self.meta = self._load_meta()
        self._init_camera()
        self.ref_indices = self._resolve_reference_indices(ref_indices)

        if self.split == "train":
            self._prepare_training_buffers()
        elif self.split in ("test_train", "test_train2"):
            self._prepare_test_poses()

    def _load_meta(self):
        candidates = []
        if self.split == "train" or self.split in ("test_train", "test_train2"):
            candidates.append("transforms_train.json")
        elif self.split == "val":
            candidates.extend(["transforms_val.json", "transforms_test.json", "transforms_train.json"])
        else:
            candidates.extend([
                "transforms_{}.json".format(self.split),
                "transforms_test.json",
                "transforms_train.json",
            ])

        for name in candidates:
            path = os.path.join(self.root_dir, name)
            if os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)
        raise FileNotFoundError("변환 행렬 JSON을 찾지 못했습니다: {}".format(candidates))

    def _init_camera(self):
        width, height = self.img_wh
        base_width = float(self.meta.get("w", 800))
        self.focal = 0.5 * base_width / np.tan(0.5 * self.meta["camera_angle_x"])
        self.focal *= float(width) / base_width
        self.K = torch.tensor(
            [[self.focal, 0.0, (width - 1.0) / 2.0],
             [0.0, self.focal, (height - 1.0) / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        self.near = 2.0
        self.far = 6.0
        self.directions = get_ray_directions(height, width, self.focal)

    def _resolve_reference_indices(self, ref_indices):
        frame_count = len(self.meta["frames"])
        if ref_indices is not None and len(ref_indices) > 0:
            refs = [int(v) for v in ref_indices]
        else:
            scene = os.path.basename(os.path.normpath(self.root_dir)).lower()
            refs = None
            for key, pair in _DEFAULT_REFERENCE_PAIRS.items():
                if key in scene:
                    refs = list(pair)
                    break
            if refs is None:
                refs = [0, frame_count // 2]

        if len(refs) != 2:
            raise ValueError("--ref_indices에는 앞/뒤 두 인덱스를 정확히 입력해야 합니다.")
        if refs[0] == refs[1]:
            raise ValueError("두 참조 인덱스는 서로 달라야 합니다.")
        for idx in refs:
            if idx < 0 or idx >= frame_count:
                raise IndexError("참조 인덱스 {}가 프레임 범위 0..{}를 벗어났습니다.".format(idx, frame_count - 1))
        return refs

    def _load_depth(self, image_path):
        basename = os.path.basename(image_path)
        stem = os.path.splitext(basename)[0]
        candidates = []
        if self.depth_type == "nerf":
            candidates.append(os.path.join(self.root_dir, "depth_nerf", stem + ".npy"))
        elif self.depth_type == "gt":
            candidates.extend([
                os.path.join(self.root_dir, "my_testset", stem + "_400.npy"),
                os.path.join(self.root_dir, "depth_gt", stem + "_400.npy"),
            ])
        else:
            candidates.extend([
                os.path.join(self.root_dir, "depth", basename + ".npy"),
                os.path.join(self.root_dir, "depth", stem + ".npy"),
            ])

        for path in candidates:
            if os.path.exists(path):
                depth = _resize_depth(np.load(path), self.img_wh)
                depth[~torch.isfinite(depth)] = 0
                depth[depth < 0] = 0
                return depth
        raise FileNotFoundError(
            "참조 깊이 파일을 찾지 못했습니다. 확인한 경로: {}".format(candidates)
        )

    def _make_rays(self, c2w):
        rays_o, rays_d = get_rays(self.directions, c2w[:3, :4])
        return torch.cat(
            [
                rays_o,
                rays_d,
                self.near * torch.ones_like(rays_o[:, :1]),
                self.far * torch.ones_like(rays_o[:, :1]),
            ],
            dim=1,
        )

    def _prepare_training_buffers(self):
        width, height = self.img_wh
        self.ref_views = []
        self.ref_depths = []
        self.ref_rays = []
        self.ref_c2ws = []
        self.poses_real = []
        self.ref_proj_mats = []

        all_rays = []
        all_rgbs = []
        all_depths = []

        for ref_idx in self.ref_indices:
            frame = self.meta["frames"][ref_idx]
            image_path = _frame_image_path(self.root_dir, frame)
            rgb_chw = _load_rgba_as_rgb(image_path, self.img_wh, self.transform)
            rgb_hwc = rgb_chw.permute(1, 2, 0).contiguous()
            depth = self._load_depth(image_path)
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            rays = self._make_rays(c2w).view(height, width, 8)

            self.ref_views.append(rgb_hwc)
            self.ref_depths.append(depth)
            self.ref_rays.append(rays)
            self.ref_c2ws.append(c2w)
            self.poses_real.append(flatten(c2w))

            proj = torch.tensor(convert(c2w.numpy()), dtype=torch.float32)
            proj[:3, :4] = torch.matmul(self.K, proj[:3, :4])
            self.ref_proj_mats.append(proj)

            for _ in range(self.repeat):
                all_rays.append(rays.view(-1, 8))
                all_rgbs.append(rgb_hwc.view(-1, 3))
                all_depths.append(depth.view(-1, 1))

        self.all_rays = torch.cat(all_rays, dim=0)
        self.all_rgbs = torch.cat(all_rgbs, dim=0)
        self.all_depth = torch.cat(all_depths, dim=0)
        nonwhite = self.all_rgbs.sum(dim=-1) < 2.999
        if not torch.any(nonwhite):
            nonwhite = torch.ones_like(nonwhite, dtype=torch.bool)
        self.nonzero_rays = self.all_rays[nonwhite]
        self.nonzero_rgbs = self.all_rgbs[nonwhite]
        self.nonzero_depth = self.all_depth[nonwhite]

        self.all_rgb_num = self.all_rays.shape[0]
        self.rgb_num = self.nonzero_rays.shape[0]
        self._prepare_projected_depth_pool()

    def _prepare_projected_depth_pool(self):
        """Create a small, guaranteed-nonempty pseudo-depth sampling pool."""
        width, height = self.img_wh
        rng = np.random.RandomState(2025)
        projected_rays = []
        projected_depths = []

        for ref_slot in range(2):
            ref_view = self.ref_views[ref_slot]
            ref_depth = self.ref_depths[ref_slot]
            ref_c2w = self.ref_c2ws[ref_slot]
            offsets = [(0.0, 0.0, 0.0)]
            for _ in range(self.pseudo_view_count - 1):
                offsets.append(tuple(rng.normal(0.0, max(1.0, self.angle / 2.0), size=3)))

            for x, y, z in offsets:
                side_c2w = rotate_3d(ref_c2w, x, y, z).float()
                rays = self._make_rays(side_c2w).view(height, width, 8)
                _, depth = forward_warp(
                    ref_view.permute(2, 0, 1).unsqueeze(0),
                    ref_depth.unsqueeze(0),
                    self.K,
                    torch.tensor(convert(ref_c2w.numpy()), dtype=torch.float32),
                    self.K,
                    torch.tensor(convert(side_c2w.numpy()), dtype=torch.float32),
                )
                depth = torch.tensor(depth, dtype=torch.float32).view(-1, 1)
                mask = torch.isfinite(depth[:, 0]) & (depth[:, 0] > 0)
                if torch.any(mask):
                    projected_rays.append(rays.view(-1, 8)[mask])
                    projected_depths.append(depth[mask])

        if not projected_rays:
            projected_rays = [self.all_rays]
            projected_depths = [self.all_depth]
        self.proj_rays_full = torch.cat(projected_rays, dim=0)
        self.proj_depths_full = torch.cat(projected_depths, dim=0)

    def _prepare_test_poses(self):
        frame = self.meta["frames"][self.ref_indices[0]]
        ref_c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        self.poses_test = [
            (rotate(angle) @ ref_c2w)[:3, :4]
            for angle in np.linspace(-180.0, 180.0, 60, endpoint=False)
        ]

    def __len__(self):
        if self.split == "train":
            return self.samples_per_epoch
        if self.split in ("test_train", "test_train2"):
            return len(self.poses_test)
        return len(self.meta["frames"])

    def _random_patch_origin(self, rng):
        width, height = self.img_wh
        span_h = (self.patch_size - 1) * self.sH + 1
        span_w = (self.patch_size - 1) * self.sW + 1
        if span_h > height or span_w > width:
            raise ValueError(
                "patch_size/stride가 이미지보다 큽니다: patch={} sH={} sW={} img_wh={}".format(
                    self.patch_size, self.sH, self.sW, self.img_wh
                )
            )
        top = rng.randint(0, height - span_h + 1)
        left = rng.randint(0, width - span_w + 1)
        return top, left

    def _slice_patch(self, tensor, top, left):
        return tensor[
            top:top + (self.patch_size - 1) * self.sH + 1:self.sH,
            left:left + (self.patch_size - 1) * self.sW + 1:self.sW,
        ]

    def __getitem__(self, idx):
        if self.split == "train":
            return self._get_train_item(idx)
        return self._get_eval_item(idx)

    def _get_train_item(self, idx):
        width, height = self.img_wh
        rng = np.random.RandomState(np.random.randint(0, 2 ** 31 - 1))
        ref_slot = int(idx % 2) if rng.rand() < 0.5 else int(rng.randint(0, 2))
        ref_view = self.ref_views[ref_slot]
        ref_depth = self.ref_depths[ref_slot]
        ref_rays = self.ref_rays[ref_slot]
        ref_c2w = self.ref_c2ws[ref_slot]

        top, left = self._random_patch_origin(rng)
        real_patch = self._slice_patch(ref_view, top, left).permute(2, 0, 1).contiguous()
        depth_ray = self._slice_patch(ref_rays, top, left).reshape(-1, 8)
        depth_gt = self._slice_patch(ref_depth, top, left).reshape(-1, 1)
        depth_ray_rgb = self._slice_patch(ref_view, top, left).reshape(-1, 3)

        num_nonwhite = self.random_ray_count - self.random_ray_count // 10
        num_all = self.random_ray_count - num_nonwhite
        idx_nonwhite = torch.from_numpy(
            rng.choice(self.rgb_num, num_nonwhite, replace=True).astype(np.int64)
        )
        idx_all = torch.from_numpy(
            rng.choice(self.all_rgb_num, num_all, replace=True).astype(np.int64)
        )
        rays = torch.cat([self.nonzero_rays[idx_nonwhite], self.all_rays[idx_all]], dim=0)
        rgbs = torch.cat([self.nonzero_rgbs[idx_nonwhite], self.all_rgbs[idx_all]], dim=0)
        depth = torch.cat([self.nonzero_depth[idx_nonwhite], self.all_depth[idx_all]], dim=0)

        proj_idx = torch.from_numpy(
            rng.choice(self.proj_rays_full.shape[0], self.random_ray_count, replace=True).astype(np.int64)
        )

        # Generate an unseen view around the currently selected front/back reference.
        x, y, z = rng.normal(0.0, max(1.0, self.angle / 2.0), size=3)
        side_c2w = rotate_3d(ref_c2w, x, y, z).float()
        side_rays = self._make_rays(side_c2w).view(height, width, 8)
        warped_rgb, warped_depth = forward_warp(
            ref_view.permute(2, 0, 1).unsqueeze(0),
            ref_depth.unsqueeze(0),
            self.K,
            torch.tensor(convert(ref_c2w.numpy()), dtype=torch.float32),
            self.K,
            torch.tensor(convert(side_c2w.numpy()), dtype=torch.float32),
        )
        warped_rgb = torch.tensor(warped_rgb, dtype=torch.float32)
        warped_depth = torch.tensor(warped_depth, dtype=torch.float32)

        # Prefer a patch with at least one valid warped depth value.
        side_top, side_left = top, left
        for _ in range(20):
            cand_top, cand_left = self._random_patch_origin(rng)
            cand_depth = self._slice_patch(warped_depth, cand_top, cand_left)
            if torch.any(cand_depth > 0):
                side_top, side_left = cand_top, cand_left
                break

        rays_full = self._slice_patch(side_rays, side_top, side_left).reshape(-1, 8)
        warp_patch = self._slice_patch(warped_rgb, side_top, side_left).permute(2, 0, 1).contiguous()
        warp_patch_depth = self._slice_patch(warped_depth, side_top, side_left).contiguous()

        side_proj = torch.tensor(convert(side_c2w.numpy()), dtype=torch.float32)
        side_proj[:3, :4] = torch.matmul(self.K, side_proj[:3, :4])

        return {
            "rays": rays,
            "rgbs": rgbs,
            "depth": depth,
            "rays_proj": self.proj_rays_full[proj_idx],
            "depth_proj": self.proj_depths_full[proj_idx],
            "real_patch": real_patch,
            "rays_full": rays_full,
            "warp_patch": warp_patch,
            "warp_patch_depth": warp_patch_depth,
            "side_proj": side_proj,
            "ref_proj": self.ref_proj_mats[ref_slot],
            "ref_depth_full": ref_depth,
            "side_coord": torch.stack([
                torch.arange(self.patch_size) * self.sH + side_top,
                torch.arange(self.patch_size) * self.sW + side_left,
            ]),
            "pose_real": self.poses_real[ref_slot],
            "pose_fake": flatten(side_c2w),
            "depth_ray": depth_ray,
            "depth_gt": depth_gt,
            "depth_ray_rgb": depth_ray_rgb,
            "ref_id": torch.tensor(ref_slot, dtype=torch.long),
        }

    def _get_eval_item(self, idx):
        width, height = self.img_wh
        if self.split in ("test_train", "test_train2"):
            c2w = self.poses_test[idx]
            rays = self._make_rays(c2w).view(-1, 8)
            return {"rays": rays, "c2w": c2w}

        frame = self.meta["frames"][idx]
        image_path = _frame_image_path(self.root_dir, frame)
        rgb = _load_rgba_as_rgb(image_path, self.img_wh, self.transform)
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)[:3, :4]
        rays = self._make_rays(c2w).view(-1, 8)
        return {
            "rays": rays,
            "rgbs": rgb.permute(1, 2, 0).reshape(height * width, 3),
            "c2w": c2w,
            "fname": frame["file_path"],
        }
