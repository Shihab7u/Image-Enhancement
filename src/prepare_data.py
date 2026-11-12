"""Prepare the original (reference) images from the raw photographs.

data/raw/*.jpg are the author's own smartphone photographs (1932x2576 px).
To keep the experiment fast they are downsampled by a factor 2.68 to
720x960 px with area interpolation and stored losslessly as PNG in
data/original/. These PNGs are the references used by run_experiment.py.
"""
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "original"
TARGET_SIZE = (720, 960)  # (width, height)

NAMES = ["forest", "waterfall_haze", "river_boat", "waterfall_canyon", "hut"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        img = cv2.imread(str(RAW / f"{name}.jpg"), cv2.IMREAD_COLOR)
        assert img is not None, f"missing data/raw/{name}.jpg"
        small = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT / f"{name}.png"), small)
        print(f"saved {name}.png  {small.shape}")


if __name__ == "__main__":
    main()
