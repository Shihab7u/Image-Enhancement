"""Run the full pipeline: degrade -> enhance -> evaluate -> figures.

Usage (from the repository root):
    python src/run_experiment.py
"""
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from enhance import METHODS, SEED, CONTRAST_FACTOR, NOISE_SIGMA, degrade
from metrics import mae, psnr, ssim

ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT / "data" / "original"
DEG = ROOT / "data" / "degraded"
RES = ROOT / "results"
IMG_OUT, FIG_OUT = RES / "images", RES / "figures"

IMAGE_NAMES = ["forest", "waterfall_haze", "river_boat", "waterfall_canyon", "hut"]  # fixed order -> fixed seeds
LABELS = {"baseline": "Baseline (degraded)", "stretch_L": "Stretch (L*)",
          "stretch_RGB": "Stretch (per-ch.)", "histeq_L": "Hist. eq. (L*)",
          "nlm_stretch_L": "NLM + stretch (L*)"}

# zoom windows (y0, x0, size) used for the crop figure
CROPS = {"forest": (430, 300, 200), "waterfall_haze": (560, 250, 200),
         "river_boat": (40, 380, 200), "hut": (520, 380, 200)}


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    for d in (DEG, IMG_OUT, FIG_OUT):
        d.mkdir(parents=True, exist_ok=True)

    rows, store = [], {}
    for k, name in enumerate(IMAGE_NAMES):
        ref = cv2.imread(str(ORIG / f"{name}.png"), cv2.IMREAD_COLOR)
        assert ref is not None, f"missing {name}.png - run src/prepare_data.py"
        deg = degrade(ref, seed=SEED + k)
        cv2.imwrite(str(DEG / f"{name}_degraded.png"), deg)

        store[name] = {"original": ref, "degraded": deg}
        for m, fn in METHODS.items():
            out = fn(deg)
            store[name][m] = out
            cv2.imwrite(str(IMG_OUT / f"{name}_{m}.png"), out)
            rows.append({"image": name, "method": m,
                         "psnr_db": round(psnr(ref, out), 2),
                         "ssim": round(ssim(ref, out), 4),
                         "mae": round(mae(ref, out), 2)})
            print(f"{name:8s} {m:14s} PSNR={rows[-1]['psnr_db']:6.2f}  "
                  f"SSIM={rows[-1]['ssim']:.4f}  MAE={rows[-1]['mae']:6.2f}")

        # methods applied to the *real* original (no reference available)
        for m in ("stretch_L", "histeq_L"):
            cv2.imwrite(str(IMG_OUT / f"{name}_REAL_{m}.png"), METHODS[m](ref))
            store[name][f"real_{m}"] = METHODS[m](ref)

    with open(RES / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---------- Figure 1: full comparison grid ----------
    cols = ["original", "degraded"] + [m for m in METHODS if m != "baseline"]
    titles = ["Original (reference)", "Degraded (baseline)"] + [LABELS[m] for m in cols[2:]]
    fig, ax = plt.subplots(len(IMAGE_NAMES), len(cols),
                           figsize=(2.0 * len(cols), 2.62 * len(IMAGE_NAMES)))
    for i, name in enumerate(IMAGE_NAMES):
        for j, c in enumerate(cols):
            a = ax[i, j]
            a.imshow(rgb(store[name][c]), vmin=0, vmax=255)  # same [0,255] scale
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(titles[j], fontsize=8)
            if j == 0:
                a.set_ylabel(name, fontsize=9)
    plt.tight_layout(pad=0.3)
    fig.savefig(FIG_OUT / "comparison_grid.png", dpi=200)
    plt.close(fig)

    # ---------- Figure 2: zoomed crops (noise / detail) ----------
    sel = list(CROPS)
    fig, ax = plt.subplots(len(sel), len(cols), figsize=(2.2 * len(cols), 2.05 * len(sel)))
    for i, name in enumerate(sel):
        y, x, s = CROPS[name]
        for j, c in enumerate(cols):
            a = ax[i, j]
            a.imshow(rgb(store[name][c][y:y + s, x:x + s]), vmin=0, vmax=255,
                     interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(titles[j], fontsize=8)
            if j == 0:
                a.set_ylabel(f"{name} (crop)", fontsize=9)
    plt.tight_layout(pad=0.3)
    fig.savefig(FIG_OUT / "crops.png", dpi=200)
    plt.close(fig)

    # ---------- Figure 3: real images, no reference ----------
    fig, ax = plt.subplots(3, len(IMAGE_NAMES), figsize=(2.0 * len(IMAGE_NAMES), 8.0))
    for j, name in enumerate(IMAGE_NAMES):
        for i, (c, t) in enumerate([("original", "Original"),
                                    ("real_stretch_L", "Stretch (L*)"),
                                    ("real_histeq_L", "Hist. eq. (L*)")]):
            a = ax[i, j]
            a.imshow(rgb(store[name][c]))
            a.set_xticks([]); a.set_yticks([])
            if j == 0:
                a.set_ylabel(t, fontsize=9)
            if i == 0:
                a.set_title(name, fontsize=9)
    plt.tight_layout(pad=0.3)
    fig.savefig(FIG_OUT / "real_no_reference.png", dpi=200)
    plt.close(fig)

    print(f"\nDegradation: c={CONTRAST_FACTOR}, sigma={NOISE_SIGMA}, seed={SEED}+k")
    print(f"Metrics written to {RES / 'metrics.csv'}")


if __name__ == "__main__":
    main()
