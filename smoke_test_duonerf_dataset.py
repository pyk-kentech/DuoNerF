"""Fast dataset-only smoke test for a prepared DuoNeRF Blender scene."""

import argparse

from datasets.blender_ray_patch_2image_rot3d import Blender_ray_patch_2image_rot3d_Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--ref_indices", nargs=2, type=int, required=True)
    parser.add_argument("--img_wh", nargs=2, type=int, default=[400, 400])
    parser.add_argument("--patch_size", type=int, default=32)
    args = parser.parse_args()

    dataset = Blender_ray_patch_2image_rot3d_Dataset(
        root_dir=args.root_dir,
        split="train",
        img_wh=args.img_wh,
        patch_size=args.patch_size,
        with_ref=True,
        load_depth=True,
        depth_type="nerf",
        sH=1,
        sW=1,
        angle=8,
        ref_indices=args.ref_indices,
        duo_samples_per_epoch=2,
        duo_random_rays=128,
        duo_pseudo_views=2,
    )
    sample = dataset[0]
    expected = {
        "rays": (128, 8),
        "rgbs": (128, 3),
        "depth": (128, 1),
        "real_patch": (3, args.patch_size, args.patch_size),
        "rays_full": (args.patch_size * args.patch_size, 8),
        "depth_ray": (args.patch_size * args.patch_size, 8),
        "depth_gt": (args.patch_size * args.patch_size, 1),
    }
    for key, shape in expected.items():
        actual = tuple(sample[key].shape)
        if actual != shape:
            raise RuntimeError("{} shape mismatch: expected {}, got {}".format(key, shape, actual))
    print("DUONERF_DATASET_SMOKE_OK")
    print("reference_indices={}".format(dataset.ref_indices))
    print("reference_ray_pool={}".format(tuple(dataset.all_rays.shape)))
    print("pseudo_depth_pool={}".format(tuple(dataset.proj_rays_full.shape)))


if __name__ == "__main__":
    main()
