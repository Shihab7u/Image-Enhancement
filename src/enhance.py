"""Degradation model and enhancement methods (OpenCV + NumPy only).

All functions take and return uint8 BGR images (OpenCV convention).
"""
import cv2
import numpy as np

# --------------------------------------------------------------------------
# Degradation (fixed, documented parameters)
# --------------------------------------------------------------------------
CONTRAST_FACTOR = 0.5   # c in  I_d = mu + c (I - mu) + n
NOISE_SIGMA = 6.0       # std of Gaussian noise, in grey levels [0, 255]
SEED = 42               # base seed; image k uses SEED + k


def degrade(img: np.ndarray, seed: int,
            c: float = CONTRAST_FACTOR, sigma: float = NOISE_SIGMA) -> np.ndarray:
    """Reduce contrast around the image mean and add i.i.d. Gaussian noise.

    I_d = clip( mu + c * (I - mu) + n ),  n ~ N(0, sigma^2), per pixel & channel.
    mu is the global mean of the image (all channels), so the mean brightness
    is preserved and only the dynamic range shrinks.
    """
    rng = np.random.default_rng(seed)
    f = img.astype(np.float64)
    mu = f.mean()
    out = mu + c * (f - mu) + rng.normal(0.0, sigma, size=f.shape)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _stretch(channel: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    """Linearly map the [p_low, p_high] percentiles of `channel` to [0, 255]."""
    f = channel.astype(np.float64)
    lo, hi = np.percentile(f, [p_low, p_high])
    if hi - lo < 1e-6:
        return channel.copy()
    out = (f - lo) * 255.0 / (hi - lo)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------
P_LOW, P_HIGH = 1.0, 99.0  # saturation percentiles for linear stretching


def baseline(img: np.ndarray) -> np.ndarray:
    """Method 1: no processing."""
    return img.copy()


def stretch_luminance(img: np.ndarray, p_low=P_LOW, p_high=P_HIGH) -> np.ndarray:
    """Method 2a: linear stretch of L* in CIE-Lab; a*, b* are left untouched.

    Hue is preserved; chroma is unchanged in absolute terms, so strongly
    stretched images can look slightly less saturated.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = _stretch(lab[..., 0], p_low, p_high)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def stretch_per_channel(img: np.ndarray, p_low=P_LOW, p_high=P_HIGH) -> np.ndarray:
    """Method 2b: independent linear stretch of B, G, R.

    Each channel gets its own gain/offset, which acts like an automatic white
    balance and can shift colours (colour casts are removed *or* introduced).
    """
    return cv2.merge([_stretch(ch, p_low, p_high) for ch in cv2.split(img)])


def hist_eq_luminance(img: np.ndarray) -> np.ndarray:
    """Method 3: global histogram equalisation of L* (CIE-Lab)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.equalizeHist(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# Non-local means parameters (method 4)
NLM_H, NLM_H_COLOR, NLM_TEMPLATE, NLM_SEARCH = 4, 4, 7, 21


def nlm_then_stretch(img: np.ndarray) -> np.ndarray:
    """Method 4: non-local-means denoising followed by luminance stretching.

    Motivation: stretching multiplies the noise by the same gain as the
    signal (~2x here), so we remove noise *before* amplifying contrast.
    Trade-off: NLM can also smooth fine texture.
    """
    den = cv2.fastNlMeansDenoisingColored(img, None, NLM_H, NLM_H_COLOR,
                                          NLM_TEMPLATE, NLM_SEARCH)
    return stretch_luminance(den)


METHODS = {
    "baseline": baseline,
    "stretch_L": stretch_luminance,
    "stretch_RGB": stretch_per_channel,
    "histeq_L": hist_eq_luminance,
    "nlm_stretch_L": nlm_then_stretch,
}
