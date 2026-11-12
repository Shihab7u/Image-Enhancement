# HW1 – Image Enhancement (Computer Vision, Fall 2026)

Classical enhancement of colour images affected by low contrast, non-uniform
illumination and noise. No deep learning is used.

## Environment
* Python 3.13 (tested with 3.13; any Python >= 3.10 should work)
* Tested versions: numpy 2.5.3, opencv-python 5.0.0, scikit-image 0.26.0, matplotlib 3.11.2

```bash
pip install -r requirements.txt
```

## Run the complete experiment
From the repository root:

```bash
python src/run_experiment.py
```

This reads `data/original/*.png`, regenerates the degraded images (fixed seed),
applies every method, writes all processed images, `results/metrics.csv`
and the report figures. Runtime: a few seconds on a laptop CPU.

(Optional) `python src/prepare_data.py` regenerates `data/original/` from the raw
photographs. The PNGs are already included, so this is not needed.

## Data and licences
Five of the author's own smartphone photographs (`data/raw/*.jpg`, 1932×2576 px),
redistributed here by the author. `src/prepare_data.py` downsamples them to
720×960 px (`cv2.INTER_AREA`) and stores them losslessly in `data/original/`;
these PNGs are the references.

| File | Condition |
|---|---|
| forest.png | dark, backlit forest canopy, sun and lens flare |
| waterfall_haze.png | haze/back-light, dark vegetation, sun rays |
| river_boat.png | atmospheric haze, low contrast on the far hillside |
| waterfall_canyon.png | strong sunlight with deep shadows in the gorge |
| hut.png | bright sky vs. darker ground and hillside, haze |

## Degradation (synthetic, reproducible)
`I_d = clip(round(mu + c * (I - mu) + n))`, with `mu` = global image mean,
`c = 0.5`, `n ~ N(0, 6^2)` i.i.d. per pixel and channel, seed `42 + k`
(k = index of the image in the order forest, waterfall_haze, river_boat,
waterfall_canyon, hut).
The originals are used **only** as references for the synthetic degradation;
in real use no clean reference is available.

## Methods (`src/enhance.py`)
1. `baseline` – degraded image, no processing
2. `stretch_L` – linear stretch of CIE-Lab L*, percentiles 1–99 → [0,255]
   (`stretch_RGB`: same stretch applied per B,G,R channel, for comparison)
3. `histeq_L` – global histogram equalisation of L* (`cv2.equalizeHist`)
4. `nlm_stretch_L` – `cv2.fastNlMeansDenoisingColored(h=4, hColor=4, 7, 21)` then `stretch_L`

## Metrics (`src/metrics.py`)
PSNR (own implementation, peak 255), SSIM (scikit-image, Gaussian σ=1.5,
averaged over channels), MAE (own implementation).

## File structure
```
data/raw/             author's photographs (full resolution)
data/original/        original images, 720x960 (references)
data/degraded/        degraded inputs
src/prepare_data.py   raw photos -> 720x960 reference PNGs
src/enhance.py        degradation + enhancement methods
src/metrics.py        PSNR, SSIM, MAE
src/run_experiment.py full pipeline
results/images/       <image>_<method>.png  (+ *_REAL_* : methods on real originals)
results/figures/      figures used in the report
results/metrics.csv   metrics for every image × method
report/               LaTeX source and PDF of the report
```
