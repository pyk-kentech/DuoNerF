# DuoNeRF implementation notes

## 구현 정의

DuoNeRF는 SinNeRF의 단일 참조 이미지 경로를 앞면/뒷면 두 참조 이미지로 확장한 버전이다.
NeRF의 MLP는 하나만 사용하며 두 참조 시점이 같은 radiance field를 공동으로 학습한다.

각 학습 반복에서는 다음 순서로 동작한다.

1. 앞면 또는 뒷면 참조 시점을 선택한다.
2. 두 참조 이미지에서 합친 RGB/깊이 광선 풀로 기본 NeRF 손실을 계산한다.
3. 선택된 참조 시점의 RGB 패치와 깊이 패치로 직접 감독한다.
4. 선택된 참조 카메라 주변에 임의의 가상 카메라를 만든다.
5. 선택된 참조 깊이를 가상 카메라로 워핑해 기하학 가짜 라벨을 만든다.
6. 가상 뷰 렌더링에 깊이 일관성, ViT 의미 일관성, 선택적으로 GAN 손실을 적용한다.
7. 2단계에서는 1단계 체크포인트의 NeRF 가중치만 불러와 미세조정한다.

## 변경 파일

- `datasets/blender_ray_patch_2image_rot3d.py` 새 파일
  - 두 참조 이미지, 깊이, 카메라 자세를 읽는다.
  - 두 참조의 광선/RGB/깊이를 하나의 학습 풀로 합친다.
  - 반복마다 하나의 참조를 골라 주변 가상 뷰를 생성한다.
- `models/duonerf.py` 새 파일
  - SinNeRF와 같은 MLP를 사용하되 현재 선택된 참조 패치의 ViT 특징을 매 반복 갱신한다.
- `datasets/__init__.py`
  - 새 데이터셋을 `blender_ray_patch_2image_rot3d`로 등록한다.
- `opt.py`
  - `--model duonerf`, `--ref_indices`, DuoNeRF 샘플링 옵션을 추가한다.
- `train.py`
  - `DuoNeRF` 모델 생성 경로를 추가한다.
- `models/sinnerf.py`
  - DuoNeRF에서는 의미 참조 특징을 현재 앞/뒤 패치에 맞춰 매 반복 갱신한다.
- `eval.py`
  - 새 데이터셋 이름과 참조 인덱스를 받도록 한다.
- `smoke_test_duonerf_dataset.py` 새 파일
  - 데이터 준비 상태와 학습 샘플 모양을 빠르게 검사한다.
- `README.md`
  - 1단계/2단계 실행 명령을 추가한다.

## 데이터 전제

두 이미지가 단순히 폴더에 존재하는 것만으로는 부족하다.
두 이미지 모두 `transforms_train.json`에 포함되어 있고 각각 올바른 카메라 외부행렬을 가져야 한다.
`--depth_type nerf`를 사용할 때는 두 이미지의 깊이 파일도 다음 위치에 있어야 한다.

```text
<root_dir>/depth_nerf/<image_stem>.npy
```

예를 들어 `./train/r_20.png`, `./train/r_70.png`를 참조한다면 다음을 사용한다.

```text
--ref_indices 20 70
```

## 빠른 검사

```bash
python smoke_test_duonerf_dataset.py \
  --root_dir ../datasets/downloaded_folder/nerf_synthetic/lego \
  --ref_indices 20 70 \
  --img_wh 400 400 \
  --patch_size 32
```

성공 기준:

```text
DUONERF_DATASET_SMOKE_OK
```

## 검증 결과

- 수정·추가 Python 전체 `compileall`: 통과
- 옵션 파서에서 DuoNeRF 모델/데이터셋/두 참조 인덱스 인식: 통과
- 32x32 합성 장면에서 두 참조 데이터셋 초기화: 통과
- 앞/뒤 RGB·깊이·광선 풀 생성: 통과
- 학습 샘플의 모든 필수 키와 tensor shape 검사: 통과
- 생성된 DuoNeRF 광선을 기존 NeRF renderer에 입력하는 최소 렌더 검사: 통과

전체 2000 epoch GPU 학습은 데이터셋과 CUDA 환경이 이 실행 공간에 없으므로 수행하지 않았다.
원본 저장소가 지정한 Python 3.7 / PyTorch 1.8 / PyTorch Lightning 0.10 환경을 사용하는 것이 가장 안전하다.
