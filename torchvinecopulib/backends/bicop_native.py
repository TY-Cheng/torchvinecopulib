from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import scipy.stats
import torch
import torch.nn.functional as F

from .common import (
    _EPS,
    _SQRT_2PI,
    _bilinear_bin_2d_unit,
    _build_pdf_buffers_2d,
    _copula_normalization_diagnostics,
    _interp_rect_2d,
    _mirror_grid_2d,
    _normal_pdf,
    _normal_ppf,
    _normalize_copula_pdf_grid,
    _silverman_bandwidth_2d,
    _smooth_2d,
)


def _regularize_bandwidth_matrix(H: torch.Tensor) -> torch.Tensor:
    H = 0.5 * (H + H.T)
    eigvals, eigvecs = torch.linalg.eigh(H)
    eigvals = eigvals.clamp_min(_EPS)
    return eigvecs @ torch.diag(eigvals) @ eigvecs.T


def _bandwidth_matrix_from_spec(
    obs: torch.Tensor,
    *,
    bandwidth: str | float | torch.Tensor,
    bandwidth_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if isinstance(bandwidth, str):
        if bandwidth not in {"silverman", "isj"}:
            raise ValueError("Unsupported copula bandwidth.")
        scales = _silverman_bandwidth_2d(obs)
        H = torch.diag((scales * float(bandwidth_scale)).square())
        config = {
            "bandwidth": bandwidth,
            "bandwidth_scale": float(bandwidth_scale),
            "bandwidth_kind": "rule",
        }
    else:
        bw = torch.as_tensor(bandwidth, device=obs.device, dtype=obs.dtype)
        if bw.ndim == 0 or bw.numel() == 1:
            scales = bw.reshape(1).repeat(2) * float(bandwidth_scale)
            H = torch.diag(scales.square())
            config = {
                "bandwidth": "custom",
                "bandwidth_scale": float(bandwidth_scale),
                "bandwidth_kind": "scalar",
            }
        elif bw.ndim == 1 and bw.numel() == 2:
            scales = bw.reshape(2) * float(bandwidth_scale)
            H = torch.diag(scales.square())
            config = {
                "bandwidth": "custom",
                "bandwidth_scale": float(bandwidth_scale),
                "bandwidth_kind": "diag",
            }
        elif bw.shape == (2, 2):
            H = _regularize_bandwidth_matrix(bw.to(dtype=obs.dtype) * float(bandwidth_scale) ** 2)
            config = {
                "bandwidth": "custom",
                "bandwidth_scale": float(bandwidth_scale),
                "bandwidth_kind": "matrix",
            }
        else:
            raise ValueError("copula bandwidth must be a scalar, length-2 vector, or 2x2 matrix.")
    return H, H.reshape(-1), config


_TAU_GRID_SQ = torch.linspace(0.0, 0.98, 50, dtype=torch.float64).square()
_BETA_BW_TABLE = torch.tensor(
    [
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
        4.69204254128018,
        3.67817865896611,
        2.99306705888,
        2.50600489245533,
        2.14070935782172,
        1.85882498366308,
        1.63542027413745,
        1.45447243427281,
        1.30516553259084,
        1.18006222840396,
        1.07369616223661,
        0.982279653155062,
        0.902786075906722,
        0.832989698639277,
        0.77116216500065,
        0.71594673568386,
        0.666226838391616,
        0.621283273241791,
        0.580285687536092,
        0.542978050804531,
        0.508272377148461,
        0.476077286848213,
        0.445778127751307,
        0.417823893972421,
        0.391328969202865,
        0.366246870917148,
        0.342361362868933,
        0.31955865730248,
        0.297652278179634,
        0.276344663543029,
        0.255870283604903,
        0.235590782152833,
        0.215830541690526,
        0.196004036954186,
        0.176279793709377,
        0.156562165388165,
        0.13363202871254,
        0.116793384758542,
        0.0963960312149928,
        0.0805978666127558,
        0.0612707894515751,
        0.0402224124093885,
        0.0294236043019451,
        0.0162264027190233,
        0.00459925701909334,
    ],
    dtype=torch.float64,
)


def _resolve_mult(
    *,
    mult: float | None = None,
) -> float:
    return 1.0 if mult is None else max(float(mult), _EPS)


def _cov_cholesky_root(z: torch.Tensor) -> torch.Tensor:
    z = z.to(dtype=torch.float64)
    eye = torch.eye(2, device=z.device, dtype=z.dtype)
    if z.shape[0] < 2:
        return eye
    cov = torch.cov(z.T)
    cov = 0.5 * (cov + cov.T) + 1e-8 * eye
    return torch.linalg.cholesky(cov)


def _serialize_tensor(x: torch.Tensor) -> list[float] | list[list[float]]:
    cpu = x.detach().cpu()
    if cpu.ndim == 1:
        return [float(v) for v in cpu.tolist()]
    return [[float(v) for v in row] for row in cpu.tolist()]


def _kendall_abs_tau(obs: torch.Tensor) -> float:
    arr = obs.detach().cpu().numpy()
    tau = float(scipy.stats.kendalltau(arr[:, 0], arr[:, 1]).statistic)
    if not math.isfinite(tau):
        tau = 0.0
    return max(abs(tau), 0.2)


def _bw_beta_auto_scalar(obs: torch.Tensor, *, mult: float) -> float:
    tau = _kendall_abs_tau(obs)
    idx = int(torch.argmin(torch.abs(_TAU_GRID_SQ - tau)).item())
    base = float(_BETA_BW_TABLE[idx].item())
    bw = base * max(int(obs.shape[0]), 2) ** (-1.0 / 3.0)
    return max(float(mult) * bw, _EPS)


def _bw_tll_auto_matrix(obs: torch.Tensor, *, degree: int, mult: float) -> torch.Tensor:
    z = _normal_ppf(obs.to(dtype=torch.float64).clamp(_EPS, 1.0 - _EPS))
    B = 3.0 * max(int(obs.shape[0]), 2) ** (-1.0 / (4.0 * degree + 2.0)) * _cov_cholesky_root(z)
    return B * float(mult)


def _validate_tll_matrix(B: torch.Tensor) -> torch.Tensor:
    B = torch.as_tensor(B, dtype=torch.float64)
    if B.shape != (2, 2):
        raise ValueError("TLL bandwidth must be a 2x2 matrix.")
    det = float(torch.det(B).item())
    if not math.isfinite(det) or det <= _EPS:
        raise ValueError("TLL bandwidth matrix must have positive determinant.")
    return B


def _loo_nn_alpha_score_1d(x: torch.Tensor, alpha: float) -> float:
    x = x.reshape(-1).to(dtype=torch.float64)
    n = int(x.numel())
    if n < 6:
        return float("inf")
    k_eff = max(2, min(n - 1, int(math.ceil(float(alpha) * n))))
    diffs = (x[:, None] - x[None, :]).abs()
    diffs.fill_diagonal_(float("inf"))
    h = diffs.kthvalue(k_eff, dim=1).values.clamp_min(_EPS)
    kernel = torch.exp(-0.5 * (diffs / h[:, None]).square()) / (_SQRT_2PI * h[:, None])
    loo = kernel.sum(dim=1) / max(n - 1, 1)
    score = -torch.log(loo.clamp_min(_EPS)).mean()
    value = float(score.item())
    return value if math.isfinite(value) else float("inf")


def _bw_tll_nn_auto(
    obs: torch.Tensor,
    *,
    degree: int,
    mult: float,
    sample_cap: int = 512,
) -> dict[str, torch.Tensor | float]:
    z = _normal_ppf(obs.to(dtype=torch.float64).clamp(_EPS, 1.0 - _EPS))
    n = int(z.shape[0])
    cov = 0.5 * (torch.cov(z.T) + torch.cov(z.T).T) + 1e-8 * torch.eye(
        2, device=z.device, dtype=z.dtype
    )
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    B = eigvecs[:, order]
    signs = torch.sign(torch.diag(B))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    B = B * signs
    scores = z @ B
    if scores.shape[0] > sample_cap:
        idx = (
            torch.linspace(
                0,
                scores.shape[0] - 1,
                steps=sample_cap,
                device=scores.device,
                dtype=scores.dtype,
            )
            .round()
            .to(dtype=torch.long)
        )
        idx = torch.unique_consecutive(idx)
        scores_sel = scores[idx]
    else:
        scores_sel = scores
    alpha_grid = torch.linspace(
        max(n ** (-1.0 / 5.0), _EPS),
        1.0,
        50,
        device=scores.device,
        dtype=scores.dtype,
    )
    alpha_vec = []
    for dim_idx in range(2):
        vals = [
            _loo_nn_alpha_score_1d(scores_sel[:, dim_idx], float(alpha.item()))
            for alpha in alpha_grid
        ]
        best = int(min(range(len(vals)), key=vals.__getitem__))
        alpha_vec.append(float(alpha_grid[best].item()))
    alpha_1 = max(alpha_vec[0], _EPS)
    kappa = torch.tensor(
        [1.0, alpha_1 / max(alpha_vec[1], _EPS)],
        device=obs.device,
        dtype=obs.dtype,
    )
    d = 2.0
    if degree == 1:
        alpha = n ** (1.0 / 5.0 - d / (4.0 + d)) * alpha_1
    else:
        alpha = n ** (1.0 / 9.0 - d / (8.0 + d)) * alpha_1
    alpha = max(float(mult) * alpha, _EPS)
    return {
        "B": B.to(device=obs.device, dtype=obs.dtype),
        "alpha": alpha,
        "kappa": kappa.to(device=obs.device, dtype=obs.dtype),
        "selector_sample_size": int(scores_sel.shape[0]),
    }


def _tll_bandwidth_from_spec(
    obs: torch.Tensor,
    *,
    bandwidth: str | float | torch.Tensor | dict[str, Any],
    mult: float,
    degree: int,
    adaptive: bool,
    nn_k: int,
) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, dict[str, Any], torch.Tensor]:
    obs = obs.to(dtype=torch.float64)
    if adaptive:
        if isinstance(bandwidth, dict):
            if not {"B", "alpha", "kappa"} <= set(bandwidth):
                raise ValueError("Adaptive TLL bandwidth must provide 'B', 'alpha', and 'kappa'.")
            B = _validate_tll_matrix(
                torch.as_tensor(bandwidth["B"], device=obs.device, dtype=obs.dtype)
            )
            alpha = max(float(bandwidth["alpha"]) * float(mult), _EPS)
            kappa = (
                torch.as_tensor(bandwidth["kappa"], device=obs.device, dtype=obs.dtype)
                .reshape(-1)
                .clamp_min(_EPS)
            )
            if kappa.numel() != 2:
                raise ValueError("Adaptive TLL bandwidth 'kappa' must have length 2.")
            summary = torch.cat(
                [B.reshape(-1), torch.tensor([alpha], device=obs.device, dtype=obs.dtype), kappa]
            )
            return (
                B,
                alpha,
                kappa,
                summary,
                {
                    "bandwidth_kind": "custom_nn",
                    "mult": float(mult),
                    "bw": {
                        "B": _serialize_tensor(B),
                        "alpha": float(alpha),
                        "kappa": _serialize_tensor(kappa),
                    },
                },
                B,
            )
        if isinstance(bandwidth, str):
            if bandwidth != "auto":
                raise ValueError(
                    "Adaptive TLL backends support bandwidth='auto' or a custom mapping {B, alpha, kappa}."
                )
            spec = _bw_tll_nn_auto(obs, degree=degree, mult=mult)
            B = _validate_tll_matrix(spec["B"])
            alpha = max(float(spec["alpha"]), _EPS)
            kappa = torch.as_tensor(spec["kappa"], device=obs.device, dtype=obs.dtype).reshape(2)
            summary = torch.cat(
                [B.reshape(-1), torch.tensor([alpha], device=obs.device, dtype=obs.dtype), kappa]
            )
            return (
                B,
                alpha,
                kappa,
                summary,
                {
                    "bandwidth_kind": "auto_tll_nn",
                    "mult": float(mult),
                    "bw": {
                        "B": _serialize_tensor(B),
                        "alpha": float(alpha),
                        "kappa": _serialize_tensor(kappa),
                    },
                    "selector_sample_size": int(spec["selector_sample_size"]),
                },
                B,
            )
        raise ValueError(
            "Adaptive TLL backends support bandwidth='auto' or a custom mapping {B, alpha, kappa}."
        )
    if isinstance(bandwidth, str):
        if bandwidth == "auto":
            B = _bw_tll_auto_matrix(obs, degree=degree, mult=mult).to(
                device=obs.device, dtype=obs.dtype
            )
            return (
                B,
                1.0,
                torch.ones(2, device=obs.device, dtype=obs.dtype),
                B.reshape(-1),
                {
                    "bandwidth_kind": "auto_tll",
                    "mult": float(mult),
                    "bw": _serialize_tensor(B),
                },
                B,
            )
        raise ValueError("TLL backends support bandwidth='auto' or a custom 2x2 matrix.")
    bw = torch.as_tensor(bandwidth, device=obs.device, dtype=obs.dtype)
    if bw.shape == (2, 2):
        B = _validate_tll_matrix(bw * float(mult))
        return (
            B,
            1.0,
            torch.ones(2, device=obs.device, dtype=obs.dtype),
            B.reshape(-1),
            {
                "bandwidth_kind": "custom_matrix",
                "mult": float(mult),
                "bw": _serialize_tensor(B),
            },
            B,
        )
    raise ValueError("TLL backends support bandwidth='auto' or a custom 2x2 matrix.")


def _beta_bandwidth_from_spec(
    obs: torch.Tensor,
    *,
    bandwidth: str | float | torch.Tensor,
    mult: float,
) -> tuple[float, torch.Tensor, dict[str, Any]]:
    if isinstance(bandwidth, str):
        if bandwidth == "auto":
            bw = _bw_beta_auto_scalar(obs, mult=mult)
            summary = torch.tensor([bw], device=obs.device, dtype=obs.dtype)
            return (
                bw,
                summary,
                {
                    "bandwidth_kind": "auto_beta",
                    "mult": float(mult),
                    "bw": float(bw),
                },
            )
        raise ValueError("beta backend supports bandwidth='auto' or a positive scalar.")
    bw_t = torch.as_tensor(bandwidth, device=obs.device, dtype=obs.dtype)
    if bw_t.ndim == 0 or bw_t.numel() == 1:
        bw = float(bw_t.reshape(-1)[0].item()) * float(mult)
    else:
        raise ValueError("beta bandwidth must be 'auto' or a positive scalar.")
    bw = max(float(bw), _EPS)
    summary = torch.tensor([bw], device=obs.device, dtype=obs.dtype)
    return (
        bw,
        summary,
        {
            "bandwidth_kind": "custom_scalar",
            "mult": float(mult),
            "bw": float(bw),
        },
    )


def _poly_features_2d(dx: torch.Tensor, dy: torch.Tensor, degree: int) -> torch.Tensor:
    ones = torch.ones_like(dx)
    if degree <= 1:
        return torch.stack([ones, dx, dy], dim=1)
    return torch.stack([ones, dx, dy, dx.square(), dx * dy, dy.square()], dim=1)


def _gaussian_raw_moments_2d(
    mu: torch.Tensor, Sigma: torch.Tensor
) -> dict[tuple[int, int], torch.Tensor]:
    m1 = mu[0]
    m2 = mu[1]
    s11 = Sigma[0, 0]
    s12 = Sigma[0, 1]
    s22 = Sigma[1, 1]
    return {
        (0, 0): torch.ones((), device=mu.device, dtype=mu.dtype),
        (1, 0): m1,
        (0, 1): m2,
        (2, 0): m1.square() + s11,
        (1, 1): m1 * m2 + s12,
        (0, 2): m2.square() + s22,
        (3, 0): m1**3 + 3.0 * m1 * s11,
        (2, 1): m1.square() * m2 + m2 * s11 + 2.0 * m1 * s12,
        (1, 2): m1 * m2.square() + m1 * s22 + 2.0 * m2 * s12,
        (0, 3): m2**3 + 3.0 * m2 * s22,
        (4, 0): m1**4 + 6.0 * m1.square() * s11 + 3.0 * s11.square(),
        (3, 1): m1**3 * m2 + 3.0 * m1 * m2 * s11 + 3.0 * m1.square() * s12 + 3.0 * s11 * s12,
        (2, 2): (
            m1.square() * m2.square()
            + m1.square() * s22
            + m2.square() * s11
            + 4.0 * m1 * m2 * s12
            + s11 * s22
            + 2.0 * s12.square()
        ),
        (1, 3): m1 * m2**3 + 3.0 * m1 * m2 * s22 + 3.0 * m2.square() * s12 + 3.0 * s22 * s12,
        (0, 4): m2**4 + 6.0 * m2.square() * s22 + 3.0 * s22.square(),
    }


def _tll_integral_moments(
    beta: torch.Tensor, degree: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if degree <= 1:
        mean = beta[1:3]
        norm = torch.exp(beta[0] + 0.5 * torch.dot(mean, mean)).clamp_min(_EPS)
        mean_feat = torch.tensor(
            [1.0, float(mean[0].item()), float(mean[1].item())],
            device=beta.device,
            dtype=beta.dtype,
        )
        second = torch.tensor(
            [
                [1.0, float(mean[0].item()), float(mean[1].item())],
                [
                    float(mean[0].item()),
                    float(1.0 + mean[0].square().item()),
                    float((mean[0] * mean[1]).item()),
                ],
                [
                    float(mean[1].item()),
                    float((mean[0] * mean[1]).item()),
                    float(1.0 + mean[1].square().item()),
                ],
            ],
            device=beta.device,
            dtype=beta.dtype,
        )
        return norm, mean_feat, second
    b = beta[1:3]
    M = torch.tensor(
        [
            [1.0 - 2.0 * float(beta[3].item()), -float(beta[4].item())],
            [-float(beta[4].item()), 1.0 - 2.0 * float(beta[5].item())],
        ],
        device=beta.device,
        dtype=beta.dtype,
    )
    if not bool(torch.all(torch.isfinite(M))):
        return None
    eigvals = torch.linalg.eigvalsh(M)
    if float(eigvals.min().item()) <= _EPS:
        return None
    Sigma = torch.linalg.inv(M)
    mu = Sigma @ b
    det = torch.det(M).clamp_min(_EPS)
    norm = torch.exp(beta[0] + 0.5 * torch.dot(b, mu)) / torch.sqrt(det)
    moments = _gaussian_raw_moments_2d(mu, Sigma)
    exponents = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    mean_feat = torch.stack([moments[e] for e in exponents], dim=0)
    second = torch.empty(6, 6, device=beta.device, dtype=beta.dtype)
    for i, e_i in enumerate(exponents):
        for j, e_j in enumerate(exponents):
            second[i, j] = moments[(e_i[0] + e_j[0], e_i[1] + e_j[1])]
    return norm.clamp_min(_EPS), mean_feat, second


def _tll_objective(
    beta: torch.Tensor,
    *,
    X: torch.Tensor,
    weights: torch.Tensor,
    n_total: int,
    degree: int,
) -> torch.Tensor:
    terms = _tll_integral_moments(beta, degree)
    if terms is None:
        return torch.tensor(float("-inf"), device=beta.device, dtype=beta.dtype)
    norm, _, _ = terms
    return weights @ (X @ beta) - float(n_total) * norm


def _tll_density_at_point(
    normalized_diffs: torch.Tensor,
    *,
    scale_prod: float,
    degree: int,
    ridge: float,
    n_total: int,
    max_iter: int = 8,
) -> torch.Tensor:
    if normalized_diffs.numel() == 0:
        return torch.tensor(_EPS, device=normalized_diffs.device, dtype=normalized_diffs.dtype)
    weights = (
        torch.exp(-0.5 * normalized_diffs.square().sum(dim=1))
        / (2.0 * math.pi * max(scale_prod, _EPS))
    ).clamp_min(_EPS)
    X = _poly_features_2d(
        dx=normalized_diffs[:, 0],
        dy=normalized_diffs[:, 1],
        degree=degree,
    )
    p = X.shape[1]
    if X.shape[0] < p + 3:
        return (weights.sum() / max(float(n_total), 1.0)).clamp_min(_EPS)
    beta = torch.zeros(p, device=X.device, dtype=X.dtype)
    beta[0] = torch.log((weights.sum() / max(float(n_total), 1.0)).clamp_min(_EPS))
    eye = torch.eye(p, device=X.device, dtype=X.dtype)
    best_obj = _tll_objective(beta, X=X, weights=weights, n_total=n_total, degree=degree)
    for _ in range(max_iter):
        terms = _tll_integral_moments(beta, degree)
        if terms is None:
            break
        norm, mean_feat, second = terms
        grad = X.T @ weights - float(n_total) * norm * mean_feat
        if float(grad.abs().max().item()) < 1e-8:
            break
        lhs = float(n_total) * norm * second + float(ridge) * eye
        try:
            delta = torch.linalg.solve(lhs, grad)
        except RuntimeError:
            delta = torch.linalg.lstsq(lhs, grad.unsqueeze(1)).solution.squeeze(1)
        accepted = False
        step = 1.0
        while step >= 1e-3:
            candidate = beta + step * delta
            cand_obj = _tll_objective(
                candidate, X=X, weights=weights, n_total=n_total, degree=degree
            )
            if (
                bool(torch.isfinite(cand_obj))
                and float(cand_obj.item()) > float(best_obj.item()) + 1e-10
            ):
                beta = candidate
                best_obj = cand_obj
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    density = torch.exp(beta[0]).clamp_min(_EPS)
    if not bool(torch.isfinite(density)):
        return (weights.sum() / max(float(n_total), 1.0)).clamp_min(_EPS)
    return density


def _tll_eval_bounds(B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    corners = torch.tensor(
        [[-4.0, -4.0], [-4.0, 4.0], [4.0, -4.0], [4.0, 4.0]],
        device=B.device,
        dtype=B.dtype,
    )
    q_corners = torch.linalg.solve(B, corners.T).T
    return q_corners.min(dim=0).values, q_corners.max(dim=0).values


def _tll_nn_local_bandwidths(
    qdata: torch.Tensor,
    qeval: torch.Tensor,
    *,
    alpha: float,
    kappa: torch.Tensor,
    nn_min_scale: float,
    nn_max_scale: float,
    nn_chunk_size: int,
) -> torch.Tensor:
    n = int(qdata.shape[0])
    k_eff = max(2, min(n - 1, int(math.ceil(float(alpha) * n))))
    scaled_data = qdata / kappa
    radii_chunks = []
    for chunk in (qeval / kappa).split(max(1, int(nn_chunk_size)), dim=0):
        d = torch.cdist(chunk, scaled_data)
        radii_chunks.append(d.kthvalue(k_eff, dim=1).values.clamp_min(_EPS))
    radii = torch.cat(radii_chunks, dim=0)
    ref = radii.median().clamp_min(_EPS)
    radii = radii.clamp(min=float(nn_min_scale) * ref, max=float(nn_max_scale) * ref)
    return radii.unsqueeze(1) * kappa.unsqueeze(0)


def _fit_tll_q_density(
    qdata: torch.Tensor,
    *,
    B: torch.Tensor,
    degree: int,
    adaptive: bool,
    alpha: float,
    kappa: torch.Tensor,
    ridge: float,
    nn_min_scale: float,
    nn_max_scale: float,
    nn_chunk_size: int,
    eval_grid_size: int = 50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_min, q_max = _tll_eval_bounds(B)
    q_axis_x = torch.linspace(
        float(q_min[0].item()),
        float(q_max[0].item()),
        eval_grid_size,
        device=qdata.device,
        dtype=qdata.dtype,
    )
    q_axis_y = torch.linspace(
        float(q_min[1].item()),
        float(q_max[1].item()),
        eval_grid_size,
        device=qdata.device,
        dtype=qdata.dtype,
    )
    qeval = torch.cartesian_prod(q_axis_x, q_axis_y)
    local_bandwidths = (
        _tll_nn_local_bandwidths(
            qdata,
            qeval,
            alpha=alpha,
            kappa=kappa,
            nn_min_scale=nn_min_scale,
            nn_max_scale=nn_max_scale,
            nn_chunk_size=nn_chunk_size,
        )
        if adaptive
        else torch.ones(qeval.shape[0], 2, device=qdata.device, dtype=qdata.dtype)
    )
    n_total = int(qdata.shape[0])
    densities = []
    p = 6 if degree > 1 else 3
    for idx in range(qeval.shape[0]):
        center = qeval[idx]
        local_scale = local_bandwidths[idx]
        u = (qdata - center) / local_scale.unsqueeze(0)
        mask = (u.abs() <= 4.5).all(dim=1)
        if int(mask.sum().item()) < p + 3:
            sq = u.square().sum(dim=1)
            keep = min(max(p + 8, 24), int(qdata.shape[0]))
            top = sq.topk(keep, largest=False).indices
            u_local = u[top]
        else:
            u_local = u[mask]
        densities.append(
            _tll_density_at_point(
                u_local,
                scale_prod=float(local_scale.prod().item()),
                degree=degree,
                ridge=ridge,
                n_total=n_total,
            )
        )
    return torch.stack(densities, dim=0).view(eval_grid_size, eval_grid_size), q_axis_x, q_axis_y


def _rect_density_to_copula(
    latent_density: torch.Tensor,
    *,
    x_axis: torch.Tensor,
    y_axis: torch.Tensor,
    num_step_grid: int,
    marginal_tol: float,
    num_iter_max: int,
    transform_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    u_axis = torch.linspace(
        _EPS, 1.0 - _EPS, num_step_grid, device=x_axis.device, dtype=x_axis.dtype
    )
    z_axis = _normal_ppf(u_axis)
    eval_points = torch.cartesian_prod(z_axis, z_axis)
    if transform_matrix is not None:
        eval_rect = torch.linalg.solve(transform_matrix, eval_points.T).T
        det = abs(float(torch.det(transform_matrix).item()))
    else:
        eval_rect = eval_points
        det = 1.0
    canonical = _interp_rect_2d(
        latent_density,
        x=eval_rect[:, 0],
        y=eval_rect[:, 1],
        x_min=float(x_axis[0].item()),
        x_max=float(x_axis[-1].item()),
        y_min=float(y_axis[0].item()),
        y_max=float(y_axis[-1].item()),
    ).view(num_step_grid, num_step_grid)
    jac = torch.outer(_normal_pdf(z_axis), _normal_pdf(z_axis)).clamp_min(_EPS)
    pdf_grid = canonical / (jac * max(det, _EPS))
    step = 1.0 / max(num_step_grid - 1, 1)
    pdf_grid = _normalize_copula_pdf_grid(
        pdf_grid,
        step=step,
        marginal_tol=marginal_tol,
        num_iter_max=num_iter_max,
    )
    cdf_grid, hfunc_l_grid, hfunc_r_grid = _build_pdf_buffers_2d(pdf_grid=pdf_grid, step=step)
    return pdf_grid, cdf_grid, hfunc_l_grid, hfunc_r_grid


def fit_tll_torch_bicop(
    obs: torch.Tensor,
    *,
    num_step_grid: int,
    bandwidth: str | float | torch.Tensor | dict[str, Any],
    mult: float | None = None,
    smoother: str,
    marginal_tol: float,
    num_iter_max: int,
    degree: int,
    ridge: float = 1e-6,
    nn_k: int = 64,
    nn_alpha: float = 0.5,
    nn_min_scale: float = 0.65,
    nn_max_scale: float = 2.5,
    nn_chunk_size: int = 512,
    adaptive: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    del smoother, nn_alpha
    obs = obs.to(dtype=torch.float64).clamp(_EPS, 1.0 - _EPS)
    mult_value = _resolve_mult(mult=mult)
    B, alpha, kappa, bandwidth_summary, config, transform_matrix = _tll_bandwidth_from_spec(
        obs,
        bandwidth=bandwidth,
        mult=mult_value,
        degree=degree,
        adaptive=adaptive,
        nn_k=nn_k,
    )
    z = _normal_ppf(obs)
    qdata = torch.linalg.solve(transform_matrix, z.T).T
    latent_density, q_axis_x, q_axis_y = _fit_tll_q_density(
        qdata,
        B=B,
        degree=degree,
        adaptive=adaptive,
        alpha=alpha,
        kappa=kappa,
        ridge=ridge,
        nn_min_scale=nn_min_scale,
        nn_max_scale=nn_max_scale,
        nn_chunk_size=nn_chunk_size,
    )
    pdf_grid, cdf_grid, hfunc_l_grid, hfunc_r_grid = _rect_density_to_copula(
        latent_density,
        x_axis=q_axis_x,
        y_axis=q_axis_y,
        num_step_grid=num_step_grid,
        marginal_tol=marginal_tol,
        num_iter_max=num_iter_max,
        transform_matrix=transform_matrix,
    )
    config = {
        **config,
        "degree": int(degree),
        "adaptive": bool(adaptive),
        "ridge": float(ridge),
        "marginal_tol": float(marginal_tol),
        "num_iter_max": int(num_iter_max),
        "num_step_grid": int(num_step_grid),
        "nn_min_scale": float(nn_min_scale),
        "nn_max_scale": float(nn_max_scale),
        "nn_chunk_size": int(nn_chunk_size),
        **_copula_normalization_diagnostics(
            pdf_grid,
            step=1.0 / max(num_step_grid - 1, 1),
            marginal_tol=marginal_tol,
        ),
    }
    return pdf_grid, cdf_grid, hfunc_l_grid, hfunc_r_grid, bandwidth_summary, config


@dataclass(frozen=True)
class _TtSelectorStats:
    obs: torch.Tensor
    z_raw: torch.Tensor
    z_std: torch.Tensor
    n: int
    pilot_b: float
    pre_rho: float
    C1: torch.Tensor
    C21: torch.Tensor
    C22: torch.Tensor
    C23: torch.Tensor
    phi40: torch.Tensor
    phi04: torch.Tensor
    phi22: torch.Tensor
    phi31: torch.Tensor
    phi13: torch.Tensor


def _tt_subsample_obs(obs: torch.Tensor, sample_cap: int) -> torch.Tensor:
    if obs.shape[0] <= sample_cap:
        return obs
    idx = torch.linspace(
        0,
        int(obs.shape[0]) - 1,
        steps=max(2, int(sample_cap)),
        device=obs.device,
        dtype=obs.dtype,
    ).round()
    idx = torch.unique_consecutive(idx.to(dtype=torch.long))
    return obs[idx]


def _tt_check_params(params: torch.Tensor, *, strict: bool) -> torch.Tensor:
    out = params.to(dtype=torch.float64).reshape(-1).clone()
    if out.numel() != 4:
        raise ValueError("TT bandwidth must be a length-4 vector (h, rho, theta1, theta2).")
    if strict:
        if float(out[0].item()) <= 0.0:
            raise ValueError("The smoothing parameter h must be positive.")
        if abs(float(out[1].item())) >= 0.9999:
            raise ValueError("The correlation parameter rho must lie in (-1, 1).")
        if float(out[2].item()) <= 0.0:
            raise ValueError("The first tapering parameter theta1 must be positive.")
        return out
    out[0] = out[0].clamp_min(_EPS)
    out[1] = out[1].clamp(min=-0.999, max=0.999)
    out[2] = out[2].clamp_min(_EPS)
    return out


def _tt_eval_density(
    *,
    eval_points: torch.Tensor,
    obs: torch.Tensor,
    params: torch.Tensor,
    chunk_size: int = 512,
) -> torch.Tensor:
    params = _tt_check_params(params, strict=False)
    h = float(params[0].item())
    rho = float(params[1].item())
    theta1 = float(params[2].item())
    theta2 = float(params[3].item())
    z_data = _normal_ppf(obs.to(dtype=torch.float64).clamp(_EPS, 1.0 - _EPS))
    s_data = z_data[:, 0]
    t_data = z_data[:, 1]
    delta_sq = (
        h**4 * (1.0 - rho**2) * (4.0 * theta1**2 - theta2**2)
        + 2.0 * h**2 * (2.0 * theta1 + rho * theta2)
        + 1.0
    )
    if not math.isfinite(delta_sq) or delta_sq <= _EPS:
        raise ValueError("Invalid tapered-transformation parameters produced a nonpositive delta.")
    delta = math.sqrt(delta_sq)
    eta_arg = -(
        (4.0 * h**2 * theta1**2 - h**2 * theta2**2 + 2.0 * theta1)
        * (s_data.square() + t_data.square())
        + (2.0 * rho * h**2 * theta2**2 - 8.0 * rho * h**2 * theta1**2 + 2.0 * theta2)
        * s_data
        * t_data
    ) / (2.0 * delta_sq)
    eta = (eta_arg.exp().mean() / delta).clamp_min(_EPS)

    eval_points = eval_points.to(dtype=torch.float64).clamp(_EPS, 1.0 - _EPS)
    out_chunks = []
    norm_const = 2.0 * math.pi * math.sqrt(max(1.0 - rho**2, _EPS))
    s_data_row = s_data.unsqueeze(0)
    t_data_row = t_data.unsqueeze(0)
    for chunk in eval_points.split(max(1, int(chunk_size)), dim=0):
        z_eval = _normal_ppf(chunk)
        s = z_eval[:, [0]]
        t = z_eval[:, [1]]
        arg1 = (s - s_data_row) / h
        arg2 = (t - t_data_row) / h
        kern = torch.exp(
            -(arg1.square() + arg2.square() - 2.0 * rho * arg1 * arg2) / (2.0 * (1.0 - rho**2))
        )
        temp = kern.mean(dim=1) / norm_const / (h**2 * float(eta.item()))
        out_chunks.append(
            temp
            * torch.exp(
                -theta1 * (s.squeeze(1).square() + t.squeeze(1).square())
                - theta2 * s.squeeze(1) * t.squeeze(1)
            )
            / (_normal_pdf(s.squeeze(1)) * _normal_pdf(t.squeeze(1))).clamp_min(_EPS)
        )
    return torch.cat(out_chunks, dim=0).clamp_min(_EPS)


def _tt_selector_stats(obs: torch.Tensor, *, pilot_b: float) -> _TtSelectorStats:
    obs = obs.to(dtype=torch.float64).clamp(_EPS, 1.0 - _EPS)
    z_raw = _normal_ppf(obs)
    scales = z_raw.std(dim=0, unbiased=True).clamp_min(_EPS)
    z_std = z_raw / scales
    S = z_std[:, 0]
    T = z_std[:, 1]
    n = int(obs.shape[0])
    if n < 3:
        raise ValueError("Tapered-transformation selectors require at least three observations.")
    pre_rho = max(-0.99, min(0.99, float((S * T).mean().item())))
    Xi = S[:, None]
    Yi = T[:, None]
    Xj = S[None, :]
    Yj = T[None, :]
    dx = (Xi - Xj) / float(pilot_b)
    dy = (Yi - Yj) / float(pilot_b)
    w = _normal_pdf(dx) * _normal_pdf(dy)
    mean_st = float((S * T).mean().item())
    B1 = 2.0 - Xi.square() - Yi.square()
    B2 = mean_st - Xi * Yi
    pseudoB = torch.stack([2.0 - S.square() - T.square(), mean_st - S * T], dim=0)
    dnorm0_sq = 1.0 / (2.0 * math.pi)

    C1 = torch.stack(
        [
            torch.stack([(B1.square() * w).sum(), (B1 * B2 * w).sum()]),
            torch.stack([(B1 * B2 * w).sum(), (B2.square() * w).sum()]),
        ],
        dim=0,
    )
    C1 = (C1 - dnorm0_sq * (pseudoB @ pseudoB.T)) / (n * (n - 1) * float(pilot_b) ** 2)

    term_x = (dx.square() - 1.0) * w
    term_y = (dy.square() - 1.0) * w
    term_xy = dx * dy * w
    C21 = (
        torch.stack([(B1 * term_x).sum(), (B2 * term_x).sum()]) + dnorm0_sq * pseudoB.sum(dim=1)
    ) / (n * (n - 1) * float(pilot_b) ** 4)
    C22 = (
        torch.stack([(B1 * term_y).sum(), (B2 * term_y).sum()]) + dnorm0_sq * pseudoB.sum(dim=1)
    ) / (n * (n - 1) * float(pilot_b) ** 4)
    C23 = (
        torch.stack([(B1 * term_xy).sum(), (B2 * term_xy).sum()]) + dnorm0_sq * pseudoB.sum(dim=1)
    ) / (n * (n - 1) * float(pilot_b) ** 4)

    phi40 = ((((dx.square()).square()) - 6.0 * dx.square() + 3.0) * w).sum() / (
        n**2 * float(pilot_b) ** 6
    )
    phi04 = ((((dy.square()).square()) - 6.0 * dy.square() + 3.0) * w).sum() / (
        n**2 * float(pilot_b) ** 6
    )
    phi22 = (((dx.square() - 1.0) * (dy.square() - 1.0)) * w).sum() / (n**2 * float(pilot_b) ** 6)
    phi31 = (((dx.square() * dx - 3.0 * dx) * dy) * w).sum() / (n**2 * float(pilot_b) ** 6)
    phi13 = (((dy.square() * dy - 3.0 * dy) * dx) * w).sum() / (n**2 * float(pilot_b) ** 6)
    return _TtSelectorStats(
        obs=obs,
        z_raw=z_raw,
        z_std=z_std,
        n=n,
        pilot_b=float(pilot_b),
        pre_rho=pre_rho,
        C1=C1,
        C21=C21,
        C22=C22,
        C23=C23,
        phi40=phi40,
        phi04=phi04,
        phi22=phi22,
        phi31=phi31,
        phi13=phi13,
    )


def _tt_solve_c1(C1: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(C1.shape[0], device=C1.device, dtype=C1.dtype)
    ridge = 1e-8
    for _ in range(6):
        try:
            return torch.linalg.solve(C1 + ridge * eye, rhs)
        except RuntimeError:
            ridge *= 10.0
    return torch.linalg.lstsq(C1 + ridge * eye, rhs).solution


def _tt_C2_C3(stats: _TtSelectorStats, rho: float) -> tuple[torch.Tensor, torch.Tensor]:
    rho_t = torch.as_tensor(float(rho), device=stats.C1.device, dtype=stats.C1.dtype)
    C2 = stats.C21 + stats.C22 + 2.0 * rho_t * stats.C23
    C3 = (
        stats.phi40
        + stats.phi04
        + (4.0 * rho_t.square() + 2.0) * stats.phi22
        + 4.0 * rho_t * (stats.phi31 + stats.phi13)
    )
    return C2, C3


def _tt_M_objective(rho: float, stats: _TtSelectorStats) -> float:
    rho = float(max(-0.999, min(0.999, rho)))
    if abs(rho) >= 0.999:
        return float("inf")
    C2, C3 = _tt_C2_C3(stats, rho)
    quad = torch.dot(C2, _tt_solve_c1(stats.C1, C2))
    val = (C3 - quad) / max(1.0 - rho**2, _EPS)
    value = float(val.item())
    if not math.isfinite(value) or value <= _EPS:
        return float("inf")
    return value


def _grid_refine_minimize(
    objective,
    *,
    lo: float,
    hi: float,
    grid_size: int,
    num_refine: int,
    log_space: bool = False,
) -> tuple[float, float]:
    lo = float(lo)
    hi = float(hi)
    best_x = lo
    best_val = float("inf")
    for _ in range(max(0, int(num_refine)) + 1):
        if log_space:
            grid = torch.exp(
                torch.linspace(
                    math.log(max(lo, _EPS)),
                    math.log(max(hi, lo + _EPS)),
                    max(3, int(grid_size)),
                    dtype=torch.float64,
                )
            )
        else:
            grid = torch.linspace(lo, hi, max(3, int(grid_size)), dtype=torch.float64)
        vals = [float(objective(float(x.item()))) for x in grid]
        idx = min(range(len(vals)), key=vals.__getitem__)
        best_x = float(grid[idx].item())
        best_val = float(vals[idx])
        if len(grid) < 3:
            break
        left_idx = max(idx - 1, 0)
        right_idx = min(idx + 1, len(grid) - 1)
        lo = float(grid[left_idx].item())
        hi = float(grid[right_idx].item())
        if left_idx == right_idx:
            break
    return best_x, best_val


def _tt_profile_part1(
    *,
    z_std: torch.Tensor,
    h: float,
    rho: float,
    theta: torch.Tensor,
) -> float:
    S = z_std[:, 0]
    T = z_std[:, 1]
    theta1 = float(theta[0].item())
    theta2 = float(theta[1].item())
    delta_sq = (
        h**4 * (1.0 - rho**2) * (4.0 * theta1**2 - theta2**2)
        + 2.0 * h**2 * (2.0 * theta1 + rho * theta2)
        + 1.0
    )
    if not math.isfinite(delta_sq) or delta_sq <= _EPS:
        return float("inf")
    delta = math.sqrt(delta_sq)
    eta_arg = -(
        (4.0 * h**2 * theta1**2 - h**2 * theta2**2 + 2.0 * theta1) * (S.square() + T.square())
        + (2.0 * rho * h**2 * theta2**2 - 8.0 * rho * h**2 * theta1**2 + 2.0 * theta2) * S * T
    ) / (2.0 * delta_sq)
    eta = float((eta_arg.exp().mean() / delta).item())
    if not math.isfinite(eta) or eta <= _EPS:
        return float("inf")
    Xi = S[:, None]
    Yi = T[:, None]
    Xj = S[None, :]
    Yj = T[None, :]
    alpha1 = (
        -2.0 * h**4 * (1.0 - rho**2) * (4.0 * theta1**2 - theta2**2)
        + 2.0 * h**2 * ((rho**2 - 3.0) * theta1 - rho * theta2)
        - 1.0
    )
    alpha2 = (
        4.0 * h**4 * (1.0 - rho**2) * (4.0 * theta1**2 - theta2**2)
        + 2.0 * h**2 * (4.0 * rho * theta1 + (3.0 * rho**2 - 1.0) * theta2)
        + 2.0 * rho
    )
    alpha3 = -2.0 * h**2 * (4.0 * rho * theta1 + (1.0 + rho**2) * theta2) - 2.0 * rho
    alpha4 = 4.0 * h**2 * ((1.0 + rho**2) * theta1 + rho * theta2) + 2.0
    expo = (
        alpha1 * (Xi.square() + Xj.square() + Yi.square() + Yj.square())
        + alpha2 * (Xi * Yi + Xj * Yj)
        + alpha3 * (Xi * Yj + Xj * Yi)
        + alpha4 * (Xi * Xj + Yi * Yj)
    ) / (4.0 * h**2 * (1.0 - rho**2) * delta_sq)
    part1 = float(expo.exp().mean().item()) / (
        4.0 * math.pi * eta**2 * h**2 * delta * math.sqrt(max(1.0 - rho**2, _EPS))
    )
    return part1 if math.isfinite(part1) else float("inf")


def _tt_profile_part2(
    *, obs: torch.Tensor, z_raw: torch.Tensor, h: float, rho: float, theta: torch.Tensor
) -> float:
    theta1 = float(theta[0].item())
    theta2 = float(theta[1].item())
    params = torch.tensor([h, rho, theta1, theta2], device=obs.device, dtype=obs.dtype)
    params = _tt_check_params(params, strict=False)
    h = float(params[0].item())
    rho = float(params[1].item())
    theta1 = float(params[2].item())
    theta2 = float(params[3].item())
    s = z_raw[:, 0]
    t = z_raw[:, 1]
    delta_sq = (
        h**4 * (1.0 - rho**2) * (4.0 * theta1**2 - theta2**2)
        + 2.0 * h**2 * (2.0 * theta1 + rho * theta2)
        + 1.0
    )
    if not math.isfinite(delta_sq) or delta_sq <= _EPS:
        return float("inf")
    delta = math.sqrt(delta_sq)
    eta_arg = -(
        (4.0 * h**2 * theta1**2 - h**2 * theta2**2 + 2.0 * theta1) * (s.square() + t.square())
        + (2.0 * rho * h**2 * theta2**2 - 8.0 * rho * h**2 * theta1**2 + 2.0 * theta2) * s * t
    ) / (2.0 * delta_sq)
    eta = float((eta_arg.exp().mean() / delta).item())
    if not math.isfinite(eta) or eta <= _EPS:
        return float("inf")
    arg1 = (s[:, None] - s[None, :]) / h
    arg2 = (t[:, None] - t[None, :]) / h
    kern = torch.exp(
        -(arg1.square() + arg2.square() - 2.0 * rho * arg1 * arg2) / (2.0 * (1.0 - rho**2))
    )
    norm_const = 2.0 * math.pi * math.sqrt(max(1.0 - rho**2, _EPS))
    diag_const = 1.0 / norm_const
    loo_mean = (kern.sum(dim=1) / norm_const - diag_const) / max(int(obs.shape[0]) - 1, 1)
    part2 = (
        loo_mean / (h**2 * eta) * torch.exp(-theta1 * (s.square() + t.square()) - theta2 * s * t)
    ).sum()
    value = float(part2.item())
    return value if math.isfinite(value) else float("inf")


def _tt_select_rho(
    stats: _TtSelectorStats, *, selector_grid_size: int, selector_num_refine: int
) -> tuple[float, float]:
    return _grid_refine_minimize(
        lambda rho: _tt_M_objective(rho, stats),
        lo=-0.95,
        hi=0.95,
        grid_size=selector_grid_size,
        num_refine=selector_num_refine,
    )


def _tt_pi_selector(
    obs: torch.Tensor,
    *,
    selector_grid_size: int,
    selector_num_refine: int,
    selector_sample_cap: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    selector_obs = _tt_subsample_obs(obs, int(selector_sample_cap))
    z0 = _normal_ppf(selector_obs)
    S = z0[:, 0] / z0[:, 0].std(unbiased=True).clamp_min(_EPS)
    T = z0[:, 1] / z0[:, 1].std(unbiased=True).clamp_min(_EPS)
    pre_rho = max(-0.99, min(0.99, float((S * T).mean().item())))
    pilot_b = (
        32.0
        * (1.0 - pre_rho**2) ** 3
        * math.sqrt(max(1.0 - pre_rho**2, _EPS))
        / max(9.0 * pre_rho**2 + 6.0, _EPS)
        / max(selector_obs.shape[0], 2)
    ) ** (1.0 / 8.0)
    stats = _tt_selector_stats(selector_obs, pilot_b=pilot_b)
    rho, rho_obj = _tt_select_rho(
        stats,
        selector_grid_size=selector_grid_size,
        selector_num_refine=selector_num_refine,
    )
    C2, C3 = _tt_C2_C3(stats, rho)
    quad = torch.dot(C2, _tt_solve_c1(stats.C1, C2))
    denom = float((C3 - quad).item()) * math.sqrt(max(1.0 - rho**2, _EPS))
    h = (1.0 / max(2.0 * math.pi * stats.n * max(denom, _EPS), _EPS)) ** (1.0 / 6.0)
    theta = -0.5 * h**2 * _tt_solve_c1(stats.C1, C2)
    params = _tt_check_params(
        torch.tensor(
            [h, rho, float(theta[0].item()), float(theta[1].item())],
            device=obs.device,
            dtype=obs.dtype,
        ),
        strict=False,
    )
    return params, {
        "bandwidth_kind": "auto_pi",
        "selector_method": "plugin",
        "selector_score": float(rho_obj),
        "selector_sample_size": int(selector_obs.shape[0]),
        "selector_grid_size": int(selector_grid_size),
        "selector_num_refine": int(selector_num_refine),
        "selector_sample_cap": int(selector_sample_cap),
    }


def _tt_cv_selector(
    obs: torch.Tensor,
    *,
    selector_grid_size: int,
    selector_num_refine: int,
    selector_sample_cap: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    selector_obs = _tt_subsample_obs(obs, int(selector_sample_cap))
    z0 = _normal_ppf(selector_obs)
    S = z0[:, 0] / z0[:, 0].std(unbiased=True).clamp_min(_EPS)
    T = z0[:, 1] / z0[:, 1].std(unbiased=True).clamp_min(_EPS)
    pre_rho = max(-0.99, min(0.99, float((S * T).mean().item())))
    a2 = 19.0 / 4.0 / math.pi**2
    a3 = (
        (48.0 * pre_rho**2 + 57.0)
        / 32.0
        / math.pi**2
        / max((pre_rho**2 - 1.0) ** 3 * math.sqrt(max(1.0 - pre_rho**2, _EPS)), _EPS)
    )
    a4 = (18.0 * pre_rho**6 + 360.0 * pre_rho**4 + 576.0 * pre_rho**2 + 171.0) / (
        256.0 * math.pi**2 * max((1.0 - pre_rho**2) ** 7, _EPS)
    )
    pilot_b = (
        6.0
        * a2
        / max(math.sqrt(max(a3**2 + 3.0 * a2 * a4, _EPS)) - a3, _EPS)
        / max(selector_obs.shape[0], 2)
    ) ** (1.0 / 8.0)
    stats = _tt_selector_stats(selector_obs, pilot_b=pilot_b)
    rho, rho_obj = _tt_select_rho(
        stats,
        selector_grid_size=selector_grid_size,
        selector_num_refine=selector_num_refine,
    )
    pi_params, _ = _tt_pi_selector(
        selector_obs,
        selector_grid_size=selector_grid_size,
        selector_num_refine=max(1, selector_num_refine - 1),
        selector_sample_cap=int(selector_obs.shape[0]),
    )
    h_center = float(pi_params[0].item())
    h_lo = max(0.02, h_center / 4.0)
    h_hi = min(1.0, max(h_lo * 1.5, h_center * 4.0))

    def _wcv(h: float) -> float:
        C2, _ = _tt_C2_C3(stats, rho)
        theta = -0.5 * h**2 * _tt_solve_c1(stats.C1, C2)
        part1 = _tt_profile_part1(z_std=stats.z_std, h=h, rho=rho, theta=theta)
        if not math.isfinite(part1):
            return float("inf")
        part2 = _tt_profile_part2(obs=stats.obs, z_raw=stats.z_raw, h=h, rho=rho, theta=theta)
        if not math.isfinite(part2):
            return float("inf")
        return part1 - 2.0 * part2 / stats.n

    h_opt, h_obj = _grid_refine_minimize(
        _wcv,
        lo=h_lo,
        hi=h_hi,
        grid_size=selector_grid_size,
        num_refine=selector_num_refine,
        log_space=True,
    )
    C2, _ = _tt_C2_C3(stats, rho)
    theta = -0.5 * h_opt**2 * _tt_solve_c1(stats.C1, C2)
    theta[0] = theta[0].clamp_min(_EPS)
    params = _tt_check_params(
        torch.tensor(
            [h_opt, rho, float(theta[0].item()), float(theta[1].item())],
            device=obs.device,
            dtype=obs.dtype,
        ),
        strict=False,
    )
    return params, {
        "bandwidth_kind": "auto_cv",
        "selector_method": "profile_cv",
        "selector_score": float(h_obj),
        "selector_rho_score": float(rho_obj),
        "selector_sample_size": int(selector_obs.shape[0]),
        "selector_grid_size": int(selector_grid_size),
        "selector_num_refine": int(selector_num_refine),
        "selector_sample_cap": int(selector_sample_cap),
    }


def _tt_params_from_spec(
    obs: torch.Tensor,
    *,
    bandwidth: str | float | torch.Tensor,
    mult: float | None = None,
    selector_kind: str,
    selector_grid_size: int,
    selector_num_refine: int,
    selector_sample_cap: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    mult_value = _resolve_mult(mult=mult)
    if isinstance(bandwidth, str):
        if bandwidth != "auto":
            raise ValueError(
                "TT backends only support bandwidth='auto' or a custom length-4 vector."
            )
        if selector_kind == "pi":
            params, config = _tt_pi_selector(
                obs,
                selector_grid_size=selector_grid_size,
                selector_num_refine=selector_num_refine,
                selector_sample_cap=selector_sample_cap,
            )
        else:
            params, config = _tt_cv_selector(
                obs,
                selector_grid_size=selector_grid_size,
                selector_num_refine=selector_num_refine,
                selector_sample_cap=selector_sample_cap,
            )
        params = params.clone()
        params[0] = params[0] * float(mult_value)
        config = {
            **config,
            "bandwidth": "auto",
            "mult": float(mult_value),
        }
        return _tt_check_params(params, strict=False), config
    params = torch.as_tensor(bandwidth, device=obs.device, dtype=obs.dtype).reshape(-1)
    params = _tt_check_params(params, strict=True)
    params = params.clone()
    params[0] = params[0] * float(mult_value)
    params = _tt_check_params(params, strict=True)
    return params, {
        "bandwidth_kind": "custom",
        "bandwidth": "custom",
        "mult": float(mult_value),
        "selector_method": "none",
        "selector_grid_size": int(selector_grid_size),
        "selector_num_refine": int(selector_num_refine),
        "selector_sample_cap": int(selector_sample_cap),
    }


def fit_tt_bicop(
    obs: torch.Tensor,
    *,
    num_step_grid: int,
    bandwidth: str | float | torch.Tensor,
    mult: float | None = None,
    marginal_tol: float,
    num_iter_max: int,
    selector_kind: str,
    selector_grid_size: int = 17,
    selector_num_refine: int = 3,
    selector_sample_cap: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    obs = obs.to(dtype=torch.float64).clamp(_EPS, 1.0 - _EPS)
    params, config = _tt_params_from_spec(
        obs,
        bandwidth=bandwidth,
        mult=mult,
        selector_kind=selector_kind,
        selector_grid_size=selector_grid_size,
        selector_num_refine=selector_num_refine,
        selector_sample_cap=selector_sample_cap,
    )
    u_axis = torch.linspace(_EPS, 1.0 - _EPS, num_step_grid, device=obs.device, dtype=obs.dtype)
    pdf_grid = _tt_eval_density(
        eval_points=torch.cartesian_prod(u_axis, u_axis),
        obs=obs,
        params=params,
    ).view(num_step_grid, num_step_grid)
    step = 1.0 / max(num_step_grid - 1, 1)
    pdf_grid = _normalize_copula_pdf_grid(
        pdf_grid,
        step=step,
        marginal_tol=marginal_tol,
        num_iter_max=num_iter_max,
    )
    cdf_grid, hfunc_l_grid, hfunc_r_grid = _build_pdf_buffers_2d(pdf_grid=pdf_grid, step=step)
    config = {
        **config,
        "tt_params": [float(x) for x in params.detach().cpu().tolist()],
        "marginal_tol": float(marginal_tol),
        "num_iter_max": int(num_iter_max),
        "num_step_grid": int(num_step_grid),
        **_copula_normalization_diagnostics(
            pdf_grid,
            step=step,
            marginal_tol=marginal_tol,
        ),
    }
    return pdf_grid, cdf_grid, hfunc_l_grid, hfunc_r_grid, params, config


def _beta_product_kernel_pdf(
    axis_x: torch.Tensor,
    axis_y: torch.Tensor,
    obs: torch.Tensor,
    *,
    h_x: float,
    h_y: float,
) -> torch.Tensor:
    u_x = axis_x.clamp(_EPS, 1.0 - _EPS).reshape(-1, 1)
    u_y = axis_y.clamp(_EPS, 1.0 - _EPS).reshape(-1, 1)
    a_x = (obs[:, [0]] / max(h_x, _EPS) + 1.0).T
    b_x = ((1.0 - obs[:, [0]]) / max(h_x, _EPS) + 1.0).T
    a_y = (obs[:, [1]] / max(h_y, _EPS) + 1.0).T
    b_y = ((1.0 - obs[:, [1]]) / max(h_y, _EPS) + 1.0).T
    log_norm_x = torch.lgamma(a_x + b_x) - torch.lgamma(a_x) - torch.lgamma(b_x)
    log_norm_y = torch.lgamma(a_y + b_y) - torch.lgamma(a_y) - torch.lgamma(b_y)
    bx = ((a_x - 1.0) * u_x.log() + (b_x - 1.0) * torch.log1p(-u_x) + log_norm_x).exp()
    by = ((a_y - 1.0) * u_y.log() + (b_y - 1.0) * torch.log1p(-u_y) + log_norm_y).exp()
    return (bx @ by.T) / max(int(obs.shape[0]), 1)


def fit_beta_bicop(
    obs: torch.Tensor,
    *,
    num_step_grid: int,
    bandwidth: str | float | torch.Tensor,
    mult: float | None = None,
    smoother: str,
    marginal_tol: float,
    num_iter_max: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    del smoother
    obs = obs.to(dtype=torch.float64).clamp(_EPS, 1.0 - _EPS)
    mult_value = _resolve_mult(mult=mult)
    bw_scalar, bandwidth_summary, config = _beta_bandwidth_from_spec(
        obs,
        bandwidth=bandwidth,
        mult=mult_value,
    )
    u_axis = torch.linspace(_EPS, 1.0 - _EPS, num_step_grid, device=obs.device, dtype=obs.dtype)
    pdf_grid = _beta_product_kernel_pdf(
        u_axis, u_axis, obs, h_x=bw_scalar, h_y=bw_scalar
    ).clamp_min(_EPS)
    step = 1.0 / max(num_step_grid - 1, 1)
    pdf_grid = _normalize_copula_pdf_grid(
        pdf_grid,
        step=step,
        marginal_tol=marginal_tol,
        num_iter_max=num_iter_max,
    )
    cdf_grid, hfunc_l_grid, hfunc_r_grid = _build_pdf_buffers_2d(pdf_grid=pdf_grid, step=step)
    config = {
        **config,
        "mult": float(mult_value),
        "smoother": "beta_kernel",
        "marginal_tol": float(marginal_tol),
        "num_iter_max": int(num_iter_max),
        "num_step_grid": int(num_step_grid),
        **_copula_normalization_diagnostics(
            pdf_grid,
            step=step,
            marginal_tol=marginal_tol,
        ),
    }
    return pdf_grid, cdf_grid, hfunc_l_grid, hfunc_r_grid, bandwidth_summary, config


def _beta_pdf(x: torch.Tensor, a: torch.Tensor | float, b: torch.Tensor | float) -> torch.Tensor:
    x = x.clamp(_EPS, 1.0 - _EPS)
    a_t = torch.as_tensor(a, device=x.device, dtype=x.dtype)
    b_t = torch.as_tensor(b, device=x.device, dtype=x.dtype)
    log_norm = torch.lgamma(a_t + b_t) - torch.lgamma(a_t) - torch.lgamma(b_t)
    return ((a_t - 1.0) * x.log() + (b_t - 1.0) * torch.log1p(-x) + log_norm).exp()


def _beta_cont_frac(
    a: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    *,
    max_iter: int = 96,
) -> torch.Tensor:
    one = torch.ones_like(x)
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = one.clone()
    d = (one - qab * x / qap).clamp_min(_EPS)
    d = 1.0 / d
    h = d.clone()
    for m in range(1, max_iter + 1):
        m_t = torch.full_like(x, float(m))
        m2_t = 2.0 * m_t
        aa = m_t * (b - m_t) * x / ((qam + m2_t) * (a + m2_t)).clamp_min(_EPS)
        d = (one + aa * d).clamp_min(_EPS)
        c = (one + aa / c.clamp_min(_EPS)).clamp_min(_EPS)
        d = 1.0 / d
        h = h * d * c
        aa = -(a + m_t) * (qab + m_t) * x / ((a + m2_t) * (qap + m2_t)).clamp_min(_EPS)
        d = (one + aa * d).clamp_min(_EPS)
        c = (one + aa / c.clamp_min(_EPS)).clamp_min(_EPS)
        d = 1.0 / d
        h = h * d * c
    return h


def _regularized_beta_inc(a: float, b: float, x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(_EPS, 1.0 - _EPS)
    a_t = torch.full_like(x, float(a))
    b_t = torch.full_like(x, float(b))
    bt = torch.exp(
        torch.lgamma(a_t + b_t)
        - torch.lgamma(a_t)
        - torch.lgamma(b_t)
        + a_t * x.log()
        + b_t * torch.log1p(-x)
    )
    threshold = (a_t + 1.0) / (a_t + b_t + 2.0)
    out = torch.empty_like(x)
    left = x < threshold
    if left.any():
        x_l = x[left]
        out[left] = bt[left] * _beta_cont_frac(a_t[left], b_t[left], x_l) / a_t[left]
    right = ~left
    if right.any():
        xr = x[right]
        out[right] = (
            1.0 - bt[right] * _beta_cont_frac(b_t[right], a_t[right], 1.0 - xr) / b_t[right]
        )
    return out.clamp(0.0, 1.0)


def _regularized_beta_inc_inv(
    a: float, b: float, p: torch.Tensor, *, max_iter: int = 56
) -> torch.Tensor:
    p = p.clamp(_EPS, 1.0 - _EPS)
    lo = torch.full_like(p, _EPS)
    hi = torch.full_like(p, 1.0 - _EPS)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        cdf_mid = _regularized_beta_inc(a, b, mid)
        lo = torch.where(cdf_mid < p, mid, lo)
        hi = torch.where(cdf_mid >= p, mid, hi)
    return 0.5 * (lo + hi)


def fit_beta_qt_bicop(
    obs: torch.Tensor,
    *,
    num_step_grid: int,
    bandwidth: str | float | torch.Tensor,
    bandwidth_scale: float,
    smoother: str,
    marginal_tol: float,
    num_iter_max: int,
    transform_shape: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    del smoother
    obs = obs.to(dtype=torch.float64).clamp(_EPS, 1.0 - _EPS)
    _, bandwidth_summary, config = _bandwidth_matrix_from_spec(
        obs,
        bandwidth=bandwidth,
        bandwidth_scale=bandwidth_scale,
    )
    if bandwidth_summary.numel() == 4:
        h_x = math.sqrt(max(float(bandwidth_summary[0].item()), _EPS))
        h_y = math.sqrt(max(float(bandwidth_summary[3].item()), _EPS))
    else:
        h_x = h_y = math.sqrt(max(float(bandwidth_summary.reshape(-1)[0].item()), _EPS))
    shape = max(float(transform_shape), 1.05)
    t_obs = _regularized_beta_inc_inv(shape, shape, obs)
    u_axis = torch.linspace(_EPS, 1.0 - _EPS, num_step_grid, device=obs.device, dtype=obs.dtype)
    t_axis = _regularized_beta_inc_inv(shape, shape, u_axis)
    t_pdf = _beta_product_kernel_pdf(t_axis, t_axis, t_obs, h_x=h_x, h_y=h_y).clamp_min(_EPS)
    jac = torch.outer(
        1.0 / _beta_pdf(t_axis, shape, shape).clamp_min(_EPS),
        1.0 / _beta_pdf(t_axis, shape, shape).clamp_min(_EPS),
    )
    pdf_grid = t_pdf * jac
    step = 1.0 / max(num_step_grid - 1, 1)
    pdf_grid = _normalize_copula_pdf_grid(
        pdf_grid,
        step=step,
        marginal_tol=marginal_tol,
        num_iter_max=num_iter_max,
    )
    cdf_grid, hfunc_l_grid, hfunc_r_grid = _build_pdf_buffers_2d(pdf_grid=pdf_grid, step=step)
    config = {
        **config,
        "transform_shape": shape,
        "smoother": "beta_quantile_transform",
        "marginal_tol": float(marginal_tol),
        "num_iter_max": int(num_iter_max),
        "num_step_grid": int(num_step_grid),
        **_copula_normalization_diagnostics(
            pdf_grid,
            step=step,
            marginal_tol=marginal_tol,
        ),
    }
    return pdf_grid, cdf_grid, hfunc_l_grid, hfunc_r_grid, bandwidth_summary, config


def _open_uniform_knots(
    num_basis: int, degree: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    num_interior = max(num_basis - degree - 1, 0)
    interior = (
        torch.linspace(0.0, 1.0, num_interior + 2, device=device, dtype=dtype)[1:-1]
        if num_interior > 0
        else torch.empty(0, device=device, dtype=dtype)
    )
    return torch.cat(
        [
            torch.zeros(degree + 1, device=device, dtype=dtype),
            interior,
            torch.ones(degree + 1, device=device, dtype=dtype),
        ]
    )


def _bspline_basis_matrix(
    x: torch.Tensor,
    *,
    num_basis: int,
    degree: int,
) -> torch.Tensor:
    knots = _open_uniform_knots(num_basis, degree, device=x.device, dtype=x.dtype)
    B = []
    for idx in range(num_basis + degree):
        left = knots[idx]
        right = knots[idx + 1]
        values = ((x >= left) & (x < right)).to(dtype=x.dtype)
        if idx == num_basis - 1:
            values = torch.where(x == 1.0, torch.ones_like(values), values)
        B.append(values)
    basis = torch.stack(B, dim=1)
    for d in range(1, degree + 1):
        cols = []
        for idx in range(num_basis + degree - d):
            left_num = x - knots[idx]
            left_den = (knots[idx + d] - knots[idx]).clamp_min(_EPS)
            right_num = knots[idx + d + 1] - x
            right_den = (knots[idx + d + 1] - knots[idx + 1]).clamp_min(_EPS)
            cols.append(
                (left_num / left_den) * basis[:, idx] + (right_num / right_den) * basis[:, idx + 1]
            )
        basis = torch.stack(cols, dim=1)
    return basis[:, :num_basis].clamp_min(0.0)


def fit_spline_pen_bicop(
    obs: torch.Tensor,
    *,
    num_step_grid: int,
    bandwidth: str | float | torch.Tensor,
    bandwidth_scale: float,
    smoother: str,
    marginal_tol: float,
    num_iter_max: int,
    num_basis: int = 17,
    penalty: float = 1e-2,
    spline_degree: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    obs = obs.to(dtype=torch.float64).clamp(_EPS, 1.0 - _EPS)
    H, bandwidth_summary, config = _bandwidth_matrix_from_spec(
        obs,
        bandwidth=bandwidth,
        bandwidth_scale=bandwidth_scale,
    )
    step = 1.0 / max(num_step_grid - 1, 1)
    hist = _bilinear_bin_2d_unit(obs, num_step_grid)
    mirrored = _mirror_grid_2d(hist)
    sigma_bins = torch.tensor(
        [
            math.sqrt(max(float(H[0, 0].item()), _EPS)) / max(step, _EPS),
            math.sqrt(max(float(H[1, 1].item()), _EPS)) / max(step, _EPS),
        ],
        device=obs.device,
        dtype=obs.dtype,
    ).clamp_min(0.5)
    smoothed = _smooth_2d(mirrored, sigma_bins=sigma_bins, smoother=smoother)
    target = smoothed[num_step_grid : 2 * num_step_grid, num_step_grid : 2 * num_step_grid]
    target = target / (target.sum().clamp_min(_EPS) * step**2)
    axis = torch.linspace(0.0, 1.0, num_step_grid, device=obs.device, dtype=obs.dtype)
    basis = _bspline_basis_matrix(axis, num_basis=num_basis, degree=spline_degree)
    design = torch.kron(basis, basis)
    response = target.clamp_min(_EPS).log().reshape(-1)
    weight = target.reshape(-1).clamp_min(_EPS)
    sw = weight.sqrt().unsqueeze(1)
    Xw = design * sw
    yw = response * sw.squeeze(1)
    eye_basis = torch.eye(num_basis, device=obs.device, dtype=obs.dtype)
    diff2 = eye_basis[2:] - 2.0 * eye_basis[1:-1] + eye_basis[:-2]
    rough = diff2.T @ diff2
    penalty_mat = torch.kron(rough, eye_basis) + torch.kron(eye_basis, rough)
    lhs = (
        Xw.T @ Xw
        + float(penalty) * penalty_mat
        + 1e-6 * torch.eye(design.shape[1], device=obs.device, dtype=obs.dtype)
    )
    rhs = Xw.T @ yw
    coef = torch.linalg.solve(lhs, rhs)
    pdf_grid = F.softplus((design @ coef).view(num_step_grid, num_step_grid)).clamp_min(_EPS)
    pdf_grid = _normalize_copula_pdf_grid(
        pdf_grid,
        step=step,
        marginal_tol=marginal_tol,
        num_iter_max=num_iter_max,
    )
    cdf_grid, hfunc_l_grid, hfunc_r_grid = _build_pdf_buffers_2d(pdf_grid=pdf_grid, step=step)
    config = {
        **config,
        "num_basis": int(num_basis),
        "penalty": float(penalty),
        "spline_degree": int(spline_degree),
        "smoother": smoother,
        "marginal_tol": float(marginal_tol),
        "num_iter_max": int(num_iter_max),
        "num_step_grid": int(num_step_grid),
        **_copula_normalization_diagnostics(
            pdf_grid,
            step=step,
            marginal_tol=marginal_tol,
        ),
    }
    return pdf_grid, cdf_grid, hfunc_l_grid, hfunc_r_grid, bandwidth_summary, config
