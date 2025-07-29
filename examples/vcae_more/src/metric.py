import numpy as np
import torch
from scipy import linalg


def compute_pairwise_distance(
    X: torch.Tensor, Y: torch.Tensor, sqrt: bool = False
) -> torch.Tensor:
    """Compute pairwise distance between rows of X and Y."""
    X = X.view(X.size(0), -1)
    Y = Y.view(Y.size(0), -1)
    X2 = (X**2).sum(1).unsqueeze(1)
    Y2 = (Y**2).sum(1).unsqueeze(0)
    dist = X2 + Y2 - 2 * X @ Y.T
    return dist.clamp(min=0).sqrt() if sqrt else dist


def compute_mmd(real: torch.Tensor, fake: torch.Tensor, sigmas=None) -> float:
    """Compute MMD between real and fake samples using RBF kernel."""
    if sigmas is None:
        sigmas = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    Mxx = compute_pairwise_distance(real, real)
    Mxy = compute_pairwise_distance(real, fake)
    Myy = compute_pairwise_distance(fake, fake)

    scale = Mxx.mean()
    mmd_score = 0.0
    for sigma in sigmas:
        gamma = 1.0 / (2 * sigma**2 * scale)
        Kxx = torch.exp(-gamma * Mxx)
        Kxy = torch.exp(-gamma * Mxy)
        Kyy = torch.exp(-gamma * Myy)
        mmd_score += (Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()).sqrt().item()
    return mmd_score


def compute_fid(real: torch.Tensor, fake: torch.Tensor) -> float:
    """Compute FID between real and fake samples."""
    real_np = real.view(real.size(0), -1).cpu().numpy()
    fake_np = fake.view(fake.size(0), -1).cpu().numpy()

    mu_r, mu_f = real_np.mean(0), fake_np.mean(0)
    sigma_r = np.cov(real_np, rowvar=False)
    sigma_f = np.cov(fake_np, rowvar=False)

    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = np.sum((mu_r - mu_f) ** 2) + np.trace(sigma_r + sigma_f - 2 * covmean)
    return float(fid_score)
