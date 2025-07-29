import numpy as np
import torch
from scipy import linalg


def mmd(real: torch.Tensor, fake: torch.Tensor, sigmas=[1e-3, 1e-2, 1e-1, 1, 10, 100]):
    """
    Differentiable MMD loss using Gaussian kernels with fixed sigmas and
    distance normalization via Mxx.mean().

    Parameters
    ----------
    real : (n, d) tensor
        Batch of real samples (features or images).
    fake : (m, d) tensor
        Batch of generated samples.
    sigmas : list of float
        Bandwidths for the RBF kernel. Defaults to wide, fixed list.

    Returns
    -------
    mmd : scalar tensor
        Differentiable scalar loss value.
    """
    real = real.view(real.size(0), -1)
    fake = fake.view(fake.size(0), -1)

    def pairwise_squared_distances(x, y):
        x_norm = (x**2).sum(dim=1, keepdim=True)
        y_norm = (y**2).sum(dim=1, keepdim=True)
        return x_norm + y_norm.T - 2.0 * x @ y.T

    Mxx = pairwise_squared_distances(real, real)
    Mxy = pairwise_squared_distances(real, fake)
    Myy = pairwise_squared_distances(fake, fake)

    # Normalization factor based on real-real distances
    scale = Mxx.mean().detach()

    mmd_total = 0.0
    for sigma in sigmas:
        denom = scale * 2.0 * sigma**2
        Kxx = torch.exp(-Mxx / denom)
        Kxy = torch.exp(-Mxy / denom)
        Kyy = torch.exp(-Myy / denom)

        mmd_total += Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()

    return mmd_total / len(sigmas)


def fid(X, Y):
    m = X.mean(0)
    m_w = Y.mean(0)
    X_np = X.numpy()
    Y_np = Y.numpy()

    C = np.cov(X_np.transpose())
    C_w = np.cov(Y_np.transpose())
    C_C_w_sqrt = linalg.sqrtm(C.dot(C_w), True).real

    score = m.dot(m) + m_w.dot(m_w) - 2 * m_w.dot(m) + np.trace(C + C_w - 2 * C_C_w_sqrt)
    return np.sqrt(score)


class Score:
    mmd = 0
    fid = 0


def compute_score(real, fake, sigmas=[1e-3, 1e-2, 1e-1, 1, 10, 100]):
    real = real.to("cpu")
    fake = fake.to("cpu")

    s = Score()
    s.mmd = np.sqrt(mmd(real, fake, sigmas).numpy())
    s.fid = fid(fake, real).numpy()

    return s
