import torch
import torch.nn.functional as F
import piq


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean absolute error (L1) between pred and target, averaged over all elements.

    Args:
        pred:   Predicted tensor of any shape.
        target: Ground-truth tensor, same shape as pred.

    Returns:
        Scalar tensor with the mean L1 loss.
    """
    return F.l1_loss(pred, target, reduction='mean')


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error (MSE) between pred and target, averaged over all elements.

    Args:
        pred:   Predicted tensor of any shape.
        target: Ground-truth tensor, same shape as pred.

    Returns:
        Scalar tensor with the mean squared error.
    """
    return F.mse_loss(pred, target, reduction='mean')


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio (PSNR) averaged over a batch.

    PSNR is computed per sample as 20 * log10(max_val / sqrt(MSE)), then
    averaged across the batch. A small epsilon (1e-8) is added under the
    square root for numerical stability when MSE is zero.

    Args:
        pred:    Predicted tensor of shape (B, ...).
        target:  Ground-truth tensor, same shape as pred.
        max_val: Maximum possible pixel/signal value (default 1.0 for [0, 1] range).

    Returns:
        Scalar tensor with the batch-mean PSNR in decibels (dB).
    """
    batch_size = pred.shape[0]
    mse = F.mse_loss(pred, target, reduction='none')
    mse_per_sample = mse.reshape(batch_size, -1).mean(dim=1)
    psnr_per_sample = 20 * torch.log10(max_val / torch.sqrt(mse_per_sample + 1e-8))
    return psnr_per_sample.mean()


def ssim_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Structural Similarity Index (SSIM) averaged over a batch.

    Uses piq.ssim with data_range=1.0, assuming inputs are in [0, 1].
    SSIM is computed per sample and then averaged across the batch.

    Args:
        pred:   Predicted tensor of shape (B, C, H, W), values in [0, 1].
        target: Ground-truth tensor, same shape as pred.

    Returns:
        Scalar tensor with the batch-mean SSIM (in [0, 1], higher is better).
    """
    ssim_per_sample = piq.ssim(pred, target, data_range=1.0, reduction='none')
    return ssim_per_sample.mean()
