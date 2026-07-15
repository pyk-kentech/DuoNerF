#!/usr/bin/env bash
set -u

cd "$(dirname "$0")"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -m compileall -q \
  prepare_duonerf_scene.py \
  run_duonerf_end_to_end.py \
  run_duonerf_from_images.py \
  smoke_test_duonerf_from_images.py \
  calib.py \
  sfm.py

python smoke_test_duonerf_from_images.py

rm -rf outputs/duonerf_complete_demo
python run_duonerf_from_images.py \
  --generate_demo \
  --work_dir outputs/duonerf_complete_demo \
  --img_wh 32 32 \
  --steps 2 \
  --rays_per_step 32 \
  --render_frames 2

echo "DUONERF_COMPLETE_VERIFICATION_OK"
