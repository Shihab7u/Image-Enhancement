"""Image-quality metrics against the clean reference."""
import numpy as np
from skimage.metrics import structural_similarity


def psnr(ref: np.ndarray, test: np.ndarray, peak: float = 255.0) -> float:
    """PSNR in dB over all pixels and channels (implemented by hand)."""
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10(peak ** 2 / mse)


def mae(ref: np.ndarray, test: np.ndarray) -> float:
    """Mean absolute error in grey levels."""
    return float(np.mean(np.abs(ref.astype(np.float64) - test.astype(np.float64))))


def ssim(ref: np.ndarray, test: np.ndarray) -> float:
    """Mean SSIM (scikit-image, Gaussian window, averaged over colour channels)."""
    return float(structural_similarity(ref, test, channel_axis=2, data_range=255,
                                       gaussian_weights=True, sigma=1.5,
                                       use_sample_covariance=False))
