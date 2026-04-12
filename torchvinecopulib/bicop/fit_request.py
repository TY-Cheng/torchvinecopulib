from __future__ import annotations

from typing import Any

import torch

from ..backends import DEFAULT_BICOP_BACKEND, normalize_bicop_backend

__all__ = ["BiCopFitRequestMixin"]


class BiCopFitRequestMixin:
    def _normalize_fit_request(
        self,
        *,
        bicop_backend: str | None,
        bicop_kwargs: dict[str, Any] | None,
        num_iter_max: int,
        bandwidth: str | float | torch.Tensor,
    ) -> tuple[str, dict[str, Any]]:
        aligned_backends = {"ttcv", "ttpi", "tll1", "tll2", "tll1nn", "tll2nn", "beta"}
        normalized_backend = normalize_bicop_backend(bicop_backend or DEFAULT_BICOP_BACKEND)
        normalized_kwargs = dict(bicop_kwargs or {})
        if normalized_backend == "tll_ref":
            normalized_kwargs.setdefault("nonparametric_method", "constant")
        else:
            default_bandwidth = (
                "auto"
                if normalized_backend in aligned_backends
                and isinstance(bandwidth, str)
                and bandwidth in {"silverman", "isj"}
                else bandwidth
            )
            normalized_kwargs.setdefault("bandwidth", default_bandwidth)
            normalized_kwargs.setdefault("num_iter_max", num_iter_max)
            if normalized_backend in {"ttcv", "ttpi"}:
                bw_val = normalized_kwargs.get("bandwidth", "auto")
                if isinstance(bw_val, str):
                    if bw_val in {"silverman", "isj"}:
                        normalized_kwargs["bandwidth"] = "auto"
                elif torch.as_tensor(bw_val).numel() != 4:
                    normalized_kwargs["bandwidth"] = "auto"
            elif normalized_backend in {"tll1", "tll2", "beta"}:
                bw_val = normalized_kwargs.get("bandwidth", "auto")
                if isinstance(bw_val, str) and bw_val in {"silverman", "isj"}:
                    normalized_kwargs["bandwidth"] = "auto"
            elif normalized_backend in {"tll1nn", "tll2nn"}:
                bw_val = normalized_kwargs.get("bandwidth", "auto")
                if isinstance(bw_val, str) and bw_val in {"silverman", "isj"}:
                    normalized_kwargs["bandwidth"] = "auto"
        return normalized_backend, normalized_kwargs
