from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="duonerf-scene-") as temporary:
        output = Path(temporary) / "scene"
        subprocess.run(
            [
                sys.executable,
                str(root / "prepare_duonerf_scene.py"),
                "--generate_demo",
                "--output_dir",
                str(output),
                "--img_wh",
                "32",
                "32",
                "--overwrite",
            ],
            check=True,
        )
        manifest = json.loads((output / "scene_manifest.json").read_text(encoding="utf-8"))
        transforms = json.loads((output / "transforms_train.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "DUONERF_SCENE_PREPARED"
        assert manifest["ref_indices"] == [0, 1]
        assert len(transforms["frames"]) == 2
        for stem in ("front", "back"):
            image = Image.open(output / "train" / f"{stem}.png")
            depth = np.load(output / "depth_nerf" / f"{stem}.npy")
            assert image.size == (32, 32)
            assert depth.shape == (32, 32)
            assert np.isfinite(depth).all()
            assert float(depth.min()) > 0.0
        print("DUONERF_RAW_IMAGE_PREP_SMOKE_OK")


if __name__ == "__main__":
    main()
