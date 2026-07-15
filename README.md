# DuoNeRF

**Runnable two-view Neural Radiance Field reconstruction from a front and back image.**

DuoNeRF extends the single-reference training path of [SinNeRF](https://arxiv.org/abs/2204.00928) into a practical two-reference pipeline. Given only a front image and a back image, the repository prepares the missing scene assets, trains a shared NeRF scene representation, saves a checkpoint, and renders an orbit-style GIF.

The project is designed to remain runnable even when the original DuoNeRF dataset, camera calibration, camera poses, and depth maps are no longer available.

> **Scope:** this repository prioritizes a complete and reproducible execution path. Automatically generated camera poses and depth maps are deterministic priors, not measured geometry. The output demonstrates the DuoNeRF workflow, but it should not be interpreted as metrically accurate 3D reconstruction.

한국어 실행 설명은 [`README_DUONERF.md`](README_DUONERF.md)에서 확인할 수 있습니다.

---

## Overview

```text
front image + back image
          │
          ▼
image normalization and foreground masks
          │
          ▼
silhouette-based depth priors
          │
          ▼
front/back camera poses and transforms_train.json
          │
          ▼
two-reference ray sampling
          │
          ▼
coarse and fine NeRF optimization
          │
          ▼
checkpoint + rendered frames + orbit GIF
```

### What the pipeline creates automatically

- Square, normalized RGBA reference images
- Foreground masks from alpha or estimated background color
- Silhouette-aware depth priors
- Front/back camera poses
- Blender-style `transforms_train.json`
- Optional SIFT/ORB two-view pose diagnostics
- Two-reference RGB and depth ray pools
- Coarse and fine NeRF checkpoints
- Orbit-style rendered frames and GIF
- JSON manifests containing the run configuration and results

---

## Key Features

### Two-reference NeRF training

The original SinNeRF path is extended from one fixed reference view to two fixed reference views. Both references contribute RGB rays and depth supervision to the same scene representation.

### No original dataset required

The user-facing pipeline accepts either:

1. a front and back image supplied by the user, or
2. an automatically generated demonstration pair.

The missing depth maps, camera transforms, and scene metadata are created by the repository.

### Modern runnable path

The main execution path uses current PyTorch directly and does not depend on the original legacy PyTorch Lightning training environment.

### Optional camera tools

- `sfm.py`: two-view feature matching and relative-pose diagnostics
- `calib.py`: checkerboard-based camera calibration

These tools are optional. The default front/back workflow remains usable without them.

---

## Repository Structure

```text
DuoNerF/
├── datasets/
│   └── blender_ray_patch_2image_rot3d.py
├── models/
│   ├── duonerf.py
│   ├── nerf.py
│   └── rendering.py
├── prepare_duonerf_scene.py
├── run_duonerf_from_images.py
├── run_duonerf_end_to_end.py
├── smoke_test_duonerf_from_images.py
├── smoke_test_duonerf_dataset.py
├── smoke_train_duonerf.py
├── verify_duonerf_complete.sh
├── sfm.py
├── calib.py
├── requirements_duonerf_modern.txt
├── README_DUONERF.md
└── train.py
```

| File | Purpose |
|---|---|
| `run_duonerf_from_images.py` | Main user-facing entry point from raw images to final outputs |
| `prepare_duonerf_scene.py` | Creates masks, depth priors, poses, metadata, and the scene directory |
| `run_duonerf_end_to_end.py` | Trains the coarse/fine NeRF models and renders output frames |
| `datasets/blender_ray_patch_2image_rot3d.py` | Loads two references and creates the training ray pools |
| `smoke_test_duonerf_from_images.py` | Verifies raw-image scene preparation |
| `verify_duonerf_complete.sh` | Runs the complete reproducibility check |
| `sfm.py` | Runs optional feature matching and relative-pose diagnostics |
| `calib.py` | Estimates camera intrinsics from checkerboard images |
| `train.py` | Preserved legacy SinNeRF/PyTorch Lightning training entry point |

---

## Requirements

Recommended environment:

- Linux or GitHub Codespaces
- Python 3.10 or newer
- CPU for smoke tests and small demonstrations
- CUDA GPU recommended for larger images and longer training

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_duonerf_modern.txt
```

The modern requirement file includes:

- PyTorch and TorchVision
- NumPy
- Pillow
- OpenCV
- ImageIO
- tqdm
- einops

---

## Quick Start: No Input Images

This command generates a deterministic front/back demonstration pair and runs the full pipeline.

```bash
source .venv/bin/activate

python run_duonerf_from_images.py \
  --generate_demo \
  --work_dir outputs/duonerf_demo \
  --img_wh 64 64 \
  --steps 20 \
  --rays_per_step 64 \
  --render_frames 12
```

Expected completion messages:

```text
DUONERF_SCENE_PREPARED
DUONERF_END_TO_END_OK
DUONERF_FROM_IMAGES_OK
```

The demonstration is intentionally small enough to run on CPU.

---

## Run with Two Real Images

Prepare one image of the front and one image of the back of the same rigid object.

```bash
source .venv/bin/activate

python run_duonerf_from_images.py \
  --front inputs/front.jpg \
  --back inputs/back.jpg \
  --work_dir outputs/my_object \
  --img_wh 128 128 \
  --steps 200 \
  --rays_per_step 256 \
  --render_frames 30 \
  --pose_mode front_back
```

### Input recommendations

- Keep the object centered in both images.
- Use a plain or transparent background when possible.
- Keep object scale similar between the two images.
- Use a rigid object rather than a deformable subject.
- Use similar lighting and focal length.
- Avoid strong motion blur and severe occlusion.

The current scene-preparation path requires a square output resolution, so both values passed to `--img_wh` must be identical.

---

## Camera Pose Modes

The `--pose_mode` option controls how the second camera pose is created.

| Mode | Behavior | Recommended use |
|---|---|---|
| `front_back` | Places the two cameras on opposite sides of the object | Default for a true front/back image pair |
| `auto` | Attempts feature-based relative pose and falls back to front/back poses | Images with some overlapping visible regions |
| `sfm` | Uses the recovered two-view pose when possible | Diagnostic experiments with sufficient feature overlap |

A true front and back pair often has little visual overlap. For that reason, `front_back` is the default and most reliable runnable option.

---

## Output Structure

A run with `--work_dir outputs/my_object` creates:

```text
outputs/my_object/
├── scene/
│   ├── source_images/                 # Present for generated demonstrations
│   ├── train/
│   │   ├── front.png
│   │   └── back.png
│   ├── depth_nerf/
│   │   ├── front.npy
│   │   └── back.npy
│   ├── diagnostics/
│   │   ├── front_mask.png
│   │   ├── back_mask.png
│   │   ├── feature_matches.png        # Created when feature matching is available
│   │   └── pose_diagnostics.json
│   ├── transforms_train.json
│   └── scene_manifest.json
├── result/
│   ├── frames/
│   ├── duonerf.pt
│   ├── duonerf_orbit.gif
│   └── summary.json
└── run_summary.json
```

### Main outputs

- `duonerf.pt`: coarse/fine NeRF weights, optimizer state, configuration, and loss history
- `duonerf_orbit.gif`: orbit-style novel-view rendering
- `summary.json`: training and artifact summary
- `scene_manifest.json`: generated scene assets and pose diagnostics

---

## One-Command Verification

Run the full repository verification:

```bash
chmod +x verify_duonerf_complete.sh
./verify_duonerf_complete.sh
```

The script performs:

1. Python syntax compilation
2. Raw-image preparation smoke test
3. Automatic demonstration-scene generation
4. Two-reference NeRF optimization
5. Checkpoint generation
6. Frame and GIF rendering

Expected final output:

```text
DUONERF_RAW_IMAGE_PREP_SMOKE_OK
DUONERF_SCENE_PREPARED
DUONERF_END_TO_END_OK
DUONERF_FROM_IMAGES_OK
DUONERF_COMPLETE_VERIFICATION_OK
```

---

## Run Individual Stages

### 1. Prepare only the scene assets

```bash
python prepare_duonerf_scene.py \
  --front inputs/front.jpg \
  --back inputs/back.jpg \
  --output_dir data/my_scene \
  --img_wh 128 128 \
  --pose_mode front_back \
  --overwrite
```

This creates the images, masks, depth priors, poses, `transforms_train.json`, and diagnostic manifests.

### 2. Train and render an already prepared scene

```bash
python run_duonerf_end_to_end.py \
  --root_dir data/my_scene \
  --ref_indices 0 1 \
  --img_wh 128 128 \
  --steps 200 \
  --rays_per_step 256 \
  --render_frames 30 \
  --output_dir outputs/my_scene_result
```

### 3. Test two-reference dataset construction

```bash
python smoke_test_duonerf_dataset.py \
  --root_dir data/my_scene \
  --ref_indices 0 1 \
  --img_wh 128 128 \
  --patch_size 16
```

### 4. Test one optimization step

```bash
python smoke_train_duonerf.py \
  --root_dir data/my_scene \
  --ref_indices 0 1 \
  --img_wh 128 128 \
  --rays_per_view 32
```

---

## Optional Two-View SfM Diagnostics

Use this only when the two images contain overlapping textures or object regions.

```bash
python sfm.py \
  --front inputs/front.jpg \
  --back inputs/back.jpg \
  --output_dir outputs/sfm_diagnostics
```

The diagnostic reports feature counts, matches, Essential Matrix inliers, and whether relative-pose recovery succeeded. Failure does not prevent the standard `front_back` pipeline from running.

---

## Optional Camera Calibration

For a controlled image-capture setup, camera intrinsics can be estimated from checkerboard images.

```bash
python calib.py \
  --images 'inputs/calibration/*.jpg' \
  --cols 10 \
  --rows 7 \
  --square_size 22 \
  --output_dir outputs/calibration
```

Calibration is not required for the default runnable demonstration.

---

## Implementation Notes

### Scene preparation

`prepare_duonerf_scene.py` converts two ordinary images into the Blender-style layout expected by the two-reference dataset loader.

When measured depth is unavailable, the script creates a smooth depth prior from the estimated object silhouette. Pixels near the silhouette boundary and pixels near the object center receive different proxy depths to provide a stable geometric signal.

### Two-reference supervision

The training loop samples an equal number of rays from both references at each optimization step. RGB reconstruction loss and smooth-L1 depth loss supervise the coarse and fine NeRF outputs.

### Rendering

After optimization, the model renders frames from the evaluation camera sequence and combines them into `duonerf_orbit.gif`.

---

## Main Command-Line Options

| Option | Default | Description |
|---|---:|---|
| `--front` | none | Front-image path |
| `--back` | none | Back-image path |
| `--generate_demo` | false | Generates a built-in reference pair |
| `--work_dir` | `outputs/duonerf_from_images` | Scene and result output directory |
| `--img_wh` | `64 64` | Square training and rendering resolution |
| `--steps` | `20` | Number of optimization steps |
| `--rays_per_step` | `64` | Number of rays sampled per optimization step |
| `--render_frames` | `12` | Number of output animation frames |
| `--pose_mode` | `front_back` | `front_back`, `auto`, or `sfm` |

Larger values usually increase runtime and memory use. A CUDA GPU is recommended before increasing resolution and training steps substantially.

---

## Limitations

This implementation deliberately provides a runnable fallback for missing original project assets. The following limitations are important:

- Two unconstrained images do not provide enough information for unique, metric 3D reconstruction.
- Automatically generated depth is a silhouette-based prior, not sensor or monocular-estimator ground truth.
- The default camera poses assume that the images show opposite sides of the object.
- Front/back images may have too little overlap for reliable classical SfM.
- Unseen side regions are weakly constrained and may contain artifacts.
- The small CPU demonstration verifies execution rather than reconstruction quality.
- Transparent, reflective, textureless, or deformable objects remain difficult.

For higher-quality reconstruction, use calibrated cameras, measured or estimated depth, additional overlapping views, segmentation masks, and longer GPU training.

---

## Legacy SinNeRF Training Path

The original SinNeRF-based `train.py` and `eval.py` files are preserved for compatibility and research comparison. The recommended runnable DuoNeRF entry point is:

```bash
python run_duonerf_from_images.py ...
```

The modern entry point is easier to reproduce on current Python and PyTorch versions, while the legacy path may require the older dependency versions defined by the original project.

---

## Reproducibility Status

The following path has been verified in GitHub Codespaces on CPU:

```text
raw or generated front/back images
→ masks and depth priors
→ camera transforms
→ two-reference ray pools
→ coarse/fine rendering
→ RGB and depth losses
→ backward pass
→ optimizer update
→ checkpoint
→ rendered frames
→ GIF
```

A successful verification produces `DUONERF_COMPLETE_VERIFICATION_OK`.

---

## Acknowledgements

This repository is based on the public SinNeRF implementation and its underlying NeRF codebase.

- SinNeRF: *Training Neural Radiance Fields on Complex Scenes from a Single Image*
- Original NeRF PyTorch Lightning codebase: `kwea123/nerf_pl`

The DuoNeRF additions in this repository include the two-reference dataset path, raw-image scene preparation, automatic fallback priors, current-PyTorch training, reproducibility tests, checkpoint generation, and orbit rendering.

---

## Citation

Please cite SinNeRF when using the original method or codebase:

```bibtex
@inproceedings{xu2022sinnerf,
  title     = {SinNeRF: Training Neural Radiance Fields on Complex Scenes from a Single Image},
  author    = {Xu, Dejia and Jiang, Yifan and Wang, Peihao and Fan, Zhiwen and Shi, Humphrey and Wang, Zhangyang},
  year      = {2022}
}
```

---

## License

See [`LICENSE`](LICENSE) for the repository license and the original project terms.
