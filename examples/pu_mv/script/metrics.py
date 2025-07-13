from itertools import combinations

import torch

from torchvinecopulib.util import kendall_tau, chatterjee_xi
# %%
# * Metrics for multivariate predictions (samples)


def winkler_score(
    y_true: torch.Tensor,
    y_samples: torch.Tensor,
    alpha: float = 0.05,
    Y_scaler: torch.Tensor = None,
) -> float:
    """
    Compute the Winkler (interval) score for multivariate predictions.
    y_true: [N, D]
    y_samples: [S, N, D]
    alpha: significance level (e.g. 0.05 for 95% interval)
    Returns the average Winkler score over all N time points.
    """
    # optionally inverse-transform
    if Y_scaler is not None:
        # y_true: [N,D]
        true_np = y_true.detach().cpu().numpy()
        y_np = y_samples.detach().cpu().numpy().reshape(-1, true_np.shape[1])
        true_inv = Y_scaler.inverse_transform(true_np)
        y_inv = Y_scaler.inverse_transform(y_np)
        y_true = torch.tensor(true_inv, device=y_true.device)
        y_samples = torch.tensor(y_inv.reshape(y_samples.shape), device=y_samples.device)
    lower = y_samples.quantile(alpha / 2, dim=0)
    upper = y_samples.quantile(1 - alpha / 2, dim=0)
    width = (upper - lower).abs()
    below = (lower - y_true).clamp(min=0)
    above = (y_true - upper).clamp(min=0)
    score = (width + (2.0 / alpha) * (below + above)).sum(dim=1)
    return score.mean().item()


def pinball_loss(
    y_true: torch.Tensor,
    y_samples: torch.Tensor,
    quantile: float,
    Y_scaler: torch.Tensor = None,
) -> float:
    """
    Compute pinball (quantile) loss at a given quantile for multivariate predictions.
    y_true: [N, D]
    y_samples: [S, N, D]
    quantile: between 0 and 1
    Returns average pinball loss over N and D.
    """
    if Y_scaler is not None:
        true_np = y_true.detach().cpu().numpy()
        y_np = y_samples.detach().cpu().numpy().reshape(-1, true_np.shape[1])
        true_inv = Y_scaler.inverse_transform(true_np)
        y_inv = Y_scaler.inverse_transform(y_np)
        y_true = torch.tensor(true_inv, device=y_true.device)
        y_samples = torch.tensor(y_inv.reshape(y_samples.shape), device=y_samples.device)
    q_hat = y_samples.quantile(quantile, dim=0)
    err = y_true - q_hat
    loss = torch.maximum((quantile - 1) * err, quantile * err)
    return loss.mean().item()


def energy_score(
    y_true: torch.Tensor,
    y_samples: torch.Tensor,
    Y_scaler: torch.Tensor = None,
) -> float:
    """
    Compute the multivariate energy score.
    y_true: [N, D]
    y_samples: [S, N, D]
    Returns average energy score over N.
    """
    if Y_scaler is not None:
        true_np = y_true.detach().cpu().numpy()
        y_np = y_samples.detach().cpu().numpy().reshape(-1, true_np.shape[1])
        true_inv = Y_scaler.inverse_transform(true_np)
        y_inv = Y_scaler.inverse_transform(y_np)
        y_true = torch.tensor(true_inv, device=y_true.device)
        y_samples = torch.tensor(y_inv.reshape(y_samples.shape), device=y_samples.device)
    dist_true = (y_samples - y_true.unsqueeze(0)).norm(dim=2)
    term1 = dist_true.mean(dim=0)
    samples = y_samples.permute(1, 0, 2)
    pair_dist = torch.cdist(samples, samples, p=2)
    term2 = 0.5 * pair_dist.mean(dim=(1, 2))
    return (term1 - term2).mean().item()


def variogram_score(
    y_true: torch.Tensor,
    y_samples: torch.Tensor,
    gamma: float = 0.5,
    Y_scaler: torch.Tensor = None,
) -> float:
    """
    Compute the multivariate variogram score with exponent gamma.
    y_true: [N, D]
    y_samples: [S, N, D]
    gamma: exponent on differences
    Returns average variogram score over N.
    """
    if Y_scaler is not None:
        true_np = y_true.detach().cpu().numpy()
        y_np = y_samples.detach().cpu().numpy().reshape(-1, true_np.shape[1])
        true_inv = Y_scaler.inverse_transform(true_np)
        y_inv = Y_scaler.inverse_transform(y_np)
        y_true = torch.tensor(true_inv, device=y_true.device)
        y_samples = torch.tensor(y_inv.reshape(y_samples.shape), device=y_samples.device)
    S, N, D = y_samples.shape
    idx = torch.triu_indices(D, D, offset=1)
    d1, d2 = idx[0], idx[1]
    true_diff = (y_true[:, d1] - y_true[:, d2]).abs().pow(gamma)
    samp_diff = (y_samples[:, :, d1] - y_samples[:, :, d2]).abs().pow(gamma)
    mean_samp = samp_diff.mean(dim=0)
    vs_pair = (mean_samp - true_diff).pow(2)
    return vs_pair.mean(dim=1).mean().item()


def mv_kendall_tau_deviation(
    y_true: torch.Tensor,
    y_samples: torch.Tensor,
    Y_scaler: torch.Tensor = None,
):
    if Y_scaler is not None:
        true_np = y_true.detach().cpu().numpy()
        y_np = y_samples.detach().cpu().numpy().reshape(-1, true_np.shape[1])
        true_inv = Y_scaler.inverse_transform(true_np)
        y_inv = Y_scaler.inverse_transform(y_np)
        y_true = torch.tensor(true_inv, device=y_true.device)
        y_samples = torch.tensor(y_inv.reshape(y_samples.shape), device=y_samples.device)

    N, D = y_true.shape
    pairs = list(combinations(range(D), 2))
    kendall_true = torch.stack(
        [
            kendall_tau(
                x=y_true[:, i],
                y=y_true[:, j],
            )[0]
            for i, j in pairs
        ]
    )
    # TDC for each predictive sample
    S = y_samples.shape[0]
    kendall_samples = []
    for s in range(S):
        samp = y_samples[s]  # [N, D]
        kendall_s = torch.stack(
            [
                kendall_tau(
                    x=samp[:, i],
                    y=samp[:, j],
                )[0]
                for i, j in pairs
            ]
        )
        kendall_samples.append(kendall_s)
    kendall_samples = torch.stack(kendall_samples, dim=0)
    return ((kendall_samples - kendall_true).abs().nanmean()).item()


def mv_chatterjee_xi_deviation(
    y_true: torch.Tensor,
    y_samples: torch.Tensor,
    Y_scaler: torch.Tensor = None,
) -> float:
    if Y_scaler is not None:
        true_np = y_true.detach().cpu().numpy()
        y_np = y_samples.detach().cpu().numpy().reshape(-1, true_np.shape[1])
        true_inv = Y_scaler.inverse_transform(true_np)
        y_inv = Y_scaler.inverse_transform(y_np)
        y_true = torch.tensor(true_inv, device=y_true.device)
        y_samples = torch.tensor(y_inv.reshape(y_samples.shape), device=y_samples.device)

    N, D = y_true.shape
    pairs = list(combinations(range(D), 2))
    chatterjee_true = torch.stack(
        [
            chatterjee_xi(
                x=y_true[:, i],
                y=y_true[:, j],
            )
            for i, j in pairs
        ]
    )
    # TDC for each predictive sample
    S = y_samples.shape[0]
    chatterjee_samples = []
    for s in range(S):
        samp = y_samples[s]  # [N, D]
        chatterjee_s = torch.stack(
            [
                chatterjee_xi(
                    x=samp[:, i],
                    y=samp[:, j],
                )
                for i, j in pairs
            ]
        )
        chatterjee_samples.append(chatterjee_s)
    chatterjee_samples = torch.stack(chatterjee_samples, dim=0)
    return ((chatterjee_samples - chatterjee_true).abs().nanmean()).item()
