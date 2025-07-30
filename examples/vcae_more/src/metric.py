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

    real = real.view(real.size(0), -1)
    fake = fake.view(fake.size(0), -1)
    Mxx = compute_pairwise_distance(real, real)
    Mxy = compute_pairwise_distance(real, fake)
    Myy = compute_pairwise_distance(fake, fake)
    if sigmas is None:
        # Use median heuristic to determine sigmas
        @torch.no_grad()
        def median_heuristic(X: torch.Tensor, Y: torch.Tensor):
            D = compute_pairwise_distance(X, Y)
            return torch.median(D[D > 0]).item()

        sigmas = [max(1e-3, median_heuristic(real, fake) / 2**i) for i in range(1, 4)]

    # scale = Mxx.mean()
    mmd_score = 0.0
    for sigma in sigmas:
        # gamma = 1.0 / (2 * sigma**2 * scale)
        gamma = 1.0 / (2 * sigma**2)
        Kxx = torch.exp(-gamma * Mxx)
        Kxy = torch.exp(-gamma * Mxy)
        Kyy = torch.exp(-gamma * Myy)
        mmd_score += (Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()).clamp(min=0).sqrt().item()
    return mmd_score


def compute_fid(real: torch.Tensor, fake: torch.Tensor) -> float:
    """Compute FID between real and fake samples."""
    real_np = real.view(real.size(0), -1).cpu().numpy()
    fake_np = fake.view(fake.size(0), -1).cpu().numpy()

    mu_r, mu_f = real_np.mean(0), fake_np.mean(0)
    sigma_r = np.cov(real_np, rowvar=False)
    sigma_f = np.cov(fake_np, rowvar=False)
    eps = 1e-6
    sigma_r += np.eye(sigma_r.shape[0]) * eps
    sigma_f += np.eye(sigma_f.shape[0]) * eps

    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(np.sum((mu_r - mu_f) ** 2) + np.trace(sigma_r + sigma_f - 2 * covmean))


def compute_metrics_from_loader(model, loader, device):
    """Compute batched metrics (MSE, MMD, FID, NLL-Vine) from a DataLoader."""
    model.eval()
    x_list, x_hat_list, z_list = [], [], []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_hat, z = model(x)
            x_list.append(x.cpu())
            x_hat_list.append(x_hat.cpu())
            z_list.append(z.cpu())
    x = torch.cat(x_list, dim=0)
    x_hat = torch.cat(x_hat_list, dim=0)
    z = torch.cat(z_list, dim=0)
    metrics = {
        "mse": torch.nn.functional.mse_loss(input=x_hat, target=x).item(),
        "mmd": compute_mmd(real=x, fake=x_hat),
        "fid": compute_fid(real=x, fake=x_hat),
        "nll_vine": model.get_neglogpdf_vine(z.to(device)).mean().item(),
    }
    return metrics
