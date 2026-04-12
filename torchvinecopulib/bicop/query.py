from __future__ import annotations

import sys
from typing import Literal

import torch

from ..backends.common import _EPS
from ..util import solve_ITP as _solve_ITP
from .diagnostics import BiCopDiagnostics

__all__ = ["BiCopQueryMixin"]


class BiCopQueryMixin:
    def _solve_itp(self, *args, **kwargs):
        solver = getattr(sys.modules.get(__package__), "solve_ITP", _solve_ITP)
        return solver(*args, **kwargs)

    def _clamp_unit(
        self,
        obs: torch.Tensor,
        *,
        eps: float,
        use_boundary_policy: bool,
    ) -> torch.Tensor:
        lower, upper = eps, 1.0 - eps
        clamped = obs.clamp(lower, upper)
        if use_boundary_policy and self.boundary_policy == "st":
            return obs + (clamped - obs).detach()
        return clamped

    def _interp(
        self,
        grid: torch.Tensor,
        obs: torch.Tensor,
        *,
        eps: float,
        use_boundary_policy: bool,
    ) -> torch.Tensor:
        idx = (
            self._clamp_unit(obs, eps=eps, use_boundary_policy=use_boundary_policy)
            / self.step_grid
        )
        i0 = idx.floor().long()
        di = idx - i0
        i1 = torch.minimum(
            i0 + 1,
            torch.full_like(input=i0, fill_value=int(self._target), device=idx.device),
        )
        g00 = grid[i0[:, 0], i0[:, 1]]
        g10 = grid[i1[:, 0], i0[:, 1]]
        g01 = grid[i0[:, 0], i1[:, 1]]
        g11 = grid[i1[:, 0], i1[:, 1]]
        return (
            g00
            + (g10 - g00) * di[:, 0]
            + (g01 - g00) * di[:, 1]
            + (g11 - g01 - g10 + g00) * di[:, 0] * di[:, 1]
        ).clamp_min(0.0)

    def _bisect_hinv(
        self,
        *,
        fixed: torch.Tensor,
        target: torch.Tensor,
        mode: Literal["l", "r"],
        num_iter: int = 64,
    ) -> torch.Tensor:
        lo = torch.zeros_like(target)
        hi = torch.ones_like(target)
        for _ in range(num_iter):
            mid = 0.5 * (lo + hi)
            values = (
                self.hfunc_l(obs=torch.hstack([fixed, mid]))
                if mode == "l"
                else self.hfunc_r(obs=torch.hstack([mid, fixed]))
            )
            move_lo = values < target
            lo = torch.where(move_lo, mid, lo)
            hi = torch.where(move_lo, hi, mid)
        return 0.5 * (lo + hi)

    def _stabilize_hinv(
        self,
        *,
        fixed: torch.Tensor,
        target: torch.Tensor,
        root: torch.Tensor,
        status: dict[str, torch.Tensor],
        mode: Literal["l", "r"],
        tol: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        used_fallback = status["used_fallback"]
        if used_fallback.any():
            refined = self._bisect_hinv(fixed=fixed, target=target, mode=mode)
            root = torch.where(used_fallback, refined, root)
        values = (
            self.hfunc_l(obs=torch.hstack([fixed, root]))
            if mode == "l"
            else self.hfunc_r(obs=torch.hstack([root, fixed]))
        )
        abs_err = (values - target).abs()
        bad = (~torch.isfinite(root)) | (~torch.isfinite(values)) | (abs_err > tol)
        if bad.any():
            # Independence fallback preserves a legal copula sample without creating a spurious mode.
            root = torch.where(bad, target, root)
            values = torch.where(bad, target, values)
            abs_err = (values - target).abs()
        root = root.clamp(min=0.0, max=1.0)
        return root, abs_err, bad

    @torch.no_grad()
    def diagnostics(self) -> BiCopDiagnostics:
        return BiCopDiagnostics(
            itp_failures_l=int(self.hinv_fallback_l),
            itp_failures_r=int(self.hinv_fallback_r),
            bisect_refinements_l=int(self.hinv_bisect_l),
            bisect_refinements_r=int(self.hinv_bisect_r),
            fallback_to_indep_l=int(self.fallback_to_indep_l),
            fallback_to_indep_r=int(self.fallback_to_indep_r),
            max_abs_hfunc_error_l=float(self.max_abs_hfunc_error_l),
            max_abs_hfunc_error_r=float(self.max_abs_hfunc_error_r),
        )

    def cdf(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return obs.prod(dim=1, keepdim=True)
        return self._interp(
            grid=self._cdf_grid,
            obs=obs,
            eps=0.0,
            use_boundary_policy=True,
        ).unsqueeze(dim=1)

    def hfunc_l(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return obs[:, [1]]
        return self._interp(
            grid=self._hfunc_l_grid,
            obs=obs,
            eps=0.0,
            use_boundary_policy=True,
        ).unsqueeze(dim=1)

    def hfunc_r(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return obs[:, [0]]
        return self._interp(
            grid=self._hfunc_r_grid,
            obs=obs,
            eps=0.0,
            use_boundary_policy=True,
        ).unsqueeze(dim=1)

    @torch.no_grad()
    def hinv_l(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return obs[:, [1]]
        u_l = self._clamp_unit(obs[:, [0]], eps=0.0, use_boundary_policy=False)
        p = self._clamp_unit(obs[:, [1]], eps=0.0, use_boundary_policy=False)
        root, status = self._solve_itp(
            fun=lambda u_r: self.hfunc_l(obs=torch.hstack([u_l, u_r])) - p,
            x_a=torch.zeros_like(p),
            x_b=torch.ones_like(p),
            failure_policy="linear",
            return_status=True,
        )
        root, abs_err, bad = self._stabilize_hinv(
            fixed=u_l,
            target=p,
            root=root,
            status=status,
            mode="l",
        )
        self.hinv_fallback_l.add_(
            status["used_fallback"].sum().to(dtype=self.hinv_fallback_l.dtype)
        )
        self.hinv_bisect_l.add_(status["used_fallback"].sum().to(dtype=self.hinv_bisect_l.dtype))
        self.fallback_to_indep_l.add_(bad.sum().to(dtype=self.fallback_to_indep_l.dtype))
        self.max_abs_hfunc_error_l.copy_(
            torch.maximum(
                self.max_abs_hfunc_error_l,
                torch.nan_to_num(abs_err, nan=0.0, posinf=float("inf"), neginf=float("inf")).max(),
            )
        )
        self.last_hinv_status_l = status
        return root

    @torch.no_grad()
    def hinv_r(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return obs[:, [0]]
        u_r = self._clamp_unit(obs[:, [1]], eps=0.0, use_boundary_policy=False)
        p = self._clamp_unit(obs[:, [0]], eps=0.0, use_boundary_policy=False)
        root, status = self._solve_itp(
            fun=lambda u_l: self.hfunc_r(obs=torch.hstack([u_l, u_r])) - p,
            x_a=torch.zeros_like(p),
            x_b=torch.ones_like(p),
            failure_policy="linear",
            return_status=True,
        )
        root, abs_err, bad = self._stabilize_hinv(
            fixed=u_r,
            target=p,
            root=root,
            status=status,
            mode="r",
        )
        self.hinv_fallback_r.add_(
            status["used_fallback"].sum().to(dtype=self.hinv_fallback_r.dtype)
        )
        self.hinv_bisect_r.add_(status["used_fallback"].sum().to(dtype=self.hinv_bisect_r.dtype))
        self.fallback_to_indep_r.add_(bad.sum().to(dtype=self.fallback_to_indep_r.dtype))
        self.max_abs_hfunc_error_r.copy_(
            torch.maximum(
                self.max_abs_hfunc_error_r,
                torch.nan_to_num(abs_err, nan=0.0, posinf=float("inf"), neginf=float("inf")).max(),
            )
        )
        self.last_hinv_status_r = status
        return root

    def pdf(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return torch.ones_like(obs[:, [0]])
        return self._interp(
            grid=self._pdf_grid,
            obs=obs,
            eps=self._EPS,
            use_boundary_policy=True,
        ).unsqueeze(dim=1)

    def log_pdf(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(device=self.device, dtype=self.dtype)
        if self.is_indep:
            return torch.zeros_like(obs[:, [0]])
        return self.pdf(obs=obs).clamp_min(_EPS).log()

    @torch.no_grad()
    def sample(
        self,
        num_sample: int = 100,
        seed: int | None = 42,
        is_sobol: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        device, dtype = self.device, self.dtype
        if is_sobol:
            obs = (
                torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)
                .draw(n=num_sample, dtype=dtype)
                .to(device=device)
            )
        else:
            if generator is None:
                generator = torch.Generator(device=device)
                if seed is not None:
                    generator.manual_seed(seed)
            obs = torch.rand(size=(num_sample, 2), dtype=dtype, device=device, generator=generator)
        if not self.is_indep:
            obs[:, [1]] = self.hinv_l(obs=obs)
        return obs
