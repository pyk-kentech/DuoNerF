# DuoNeRF: 앞·뒤 이미지 두 장에서 전체 시점 렌더링

이 구현은 옛 DuoNeRF 프로젝트의 목표를 실행 가능한 형태로 다시 만든 것입니다.
기존 데이터셋, `transforms_train.json`, 깊이 파일이 없어도 됩니다.

## 구현 범위

입력은 정면과 후면 이미지 두 장뿐입니다. 파이프라인이 자동으로 다음을 만듭니다.

1. 두 이미지를 같은 정사각형 해상도로 정규화
2. 알파 채널 또는 배경색을 이용한 전경 마스크 생성
3. 실루엣 기반 깊이 prior 생성
4. 정면/후면 카메라 자세와 Blender 형식 `transforms_train.json` 생성
5. 선택적으로 SIFT/ORB 기반 두 시점 SfM 진단
6. 하나의 공용 NeRF를 두 이미지의 RGB·깊이 광선으로 학습
7. 체크포인트와 360도 회전 GIF 저장

정확한 3차원 복원이 목표가 아니라 실행 가능한 옛 프로젝트 복원이 목표이므로, 카메라 보정 정보나 실제 깊이가 없을 때는 결정적 prior를 사용합니다.

## 입력 이미지 없이 바로 실행

```bash
cd /workspaces/DuoNerF
source .venv/bin/activate

python run_duonerf_from_images.py \
  --generate_demo \
  --work_dir outputs/duonerf_complete_demo \
  --img_wh 64 64 \
  --steps 20 \
  --rays_per_step 64 \
  --render_frames 12
```

성공 문구:

```text
DUONERF_SCENE_PREPARED
DUONERF_END_TO_END_OK
DUONERF_FROM_IMAGES_OK
```

결과:

```text
outputs/duonerf_complete_demo/
├── scene/
│   ├── train/front.png
│   ├── train/back.png
│   ├── depth_nerf/front.npy
│   ├── depth_nerf/back.npy
│   ├── transforms_train.json
│   └── scene_manifest.json
└── result/
    ├── duonerf.pt
    ├── duonerf_orbit.gif
    ├── frames/
    └── summary.json
```

## 실제 이미지 두 장으로 실행

```bash
python run_duonerf_from_images.py \
  --front /경로/front.jpg \
  --back /경로/back.jpg \
  --work_dir outputs/my_object \
  --img_wh 128 128 \
  --steps 200 \
  --rays_per_step 256 \
  --render_frames 30
```

두 사진이 정확히 앞·뒤라 겹치는 특징이 적을 수 있으므로 기본값은 `--pose_mode front_back`입니다. 일부 영역이 겹치는 두 사진이라면 `--pose_mode auto` 또는 `--pose_mode sfm`을 사용할 수 있습니다.

## 개별 구성요소

장면 준비만 실행:

```bash
python prepare_duonerf_scene.py \
  --front front.jpg \
  --back back.jpg \
  --output_dir data/my_scene \
  --img_wh 128 128 \
  --overwrite
```

두 시점 특징 매칭 진단:

```bash
python sfm.py \
  --front front.jpg \
  --back back.jpg \
  --output_dir outputs/sfm
```

체커보드 카메라 보정:

```bash
python calib.py \
  --images 'imgs_calib/*.jpg' \
  --cols 10 \
  --rows 7 \
  --square_size 22 \
  --output_dir outputs/calibration
```

## 빠른 검증

```bash
python smoke_test_duonerf_from_images.py
```

성공 문구:

```text
DUONERF_RAW_IMAGE_PREP_SMOKE_OK
```
