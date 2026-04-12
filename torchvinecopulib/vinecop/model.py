from __future__ import annotations

from itertools import combinations
from pprint import pformat
from textwrap import indent
from typing import Literal

import torch

from ..backends import API_VERSION, DEFAULT_BICOP_BACKEND
from ..bicop import BiCop
from .artifact import VineBuildArtifact
from .builder import VineCopBuilderMixin
from .engine import VineCopEngine
from .plot import VineCopPlotMixin
from .state import VineCopStateMixin
from .structure import VineCopStructureMixin

__all__ = ["VineCop"]


class VineCop(
    VineCopBuilderMixin,
    VineCopStateMixin,
    VineCopStructureMixin,
    VineCopPlotMixin,
    torch.nn.Module,
):
    def __init__(
        self,
        num_dim: int,
        is_cop_scale: bool = False,
        num_step_grid: int = 128,
        boundary_policy: Literal["hard", "st"] = "hard",
    ) -> None:
        """Initialize a VineCop object.

        Args:
            num_dim (int): number of dimensions.
            is_cop_scale (bool, optional): if True, the marginals are assumed in copula scale [0,1]. Otherwise, the marginals are fitted via KDE. Defaults to False.
            num_step_grid (int, optional): Grid resolution (power of 2) passed to each BiCop. Defaults to 128.
            boundary_policy (Literal["hard", "st"], optional): Query-time boundary handling policy
                propagated to all underlying pair-copula modules. Defaults to "hard".

        Attributes:
            num_dim (int):
            is_cop_scale (bool):
            num_step_grid (int):
            marginals (torch.nn.ModuleList): list of marginals for each dimension.
            bicops (torch.nn.ModuleDict): dictionary of BiCop
            struct_bcp (dict): BiCop structures on condition-ed/ing sets, independence, parents.
            struct_obs (list): Pseudo-obs structure per vine level.
            tree_bidep (list): Learned edge weights per vine level.
            sample_order (tuple): Sampling order for inverse Rosenblatt transform.
            num_obs (torch.Tensor): number of observations.
        """
        super().__init__()
        self.num_dim = num_dim
        self.is_cop_scale = is_cop_scale
        self.boundary_policy: Literal["hard", "st"] = boundary_policy
        self.marginals = torch.nn.ModuleList([None] * num_dim)
        self.bicops = torch.nn.ModuleDict()
        self.struct_bcp = {}
        self.struct_obs = [{} for _ in range(num_dim)]
        for i in range(num_dim):
            self.struct_obs[0][(i,)] = ""
        for i, j in combinations(range(num_dim), 2):
            # * num_bicop = num_dim * (num_dim - 1) // 2
            # NOTE i < j by itertools.combinations;
            # ! ModuleDict key must be str
            cond_ed = f"{i},{j}"
            self.bicops[cond_ed] = BiCop(
                num_step_grid=num_step_grid,
                boundary_policy=boundary_policy,
            )
            self.struct_bcp[cond_ed] = dict(
                # * cond_ed, cond_ing (now empty) of a bicop
                cond_ed=(i, j),
                cond_ing=tuple(),
                is_indep=True,
                # ! left parent cond_ed str
                left=None,
                # ! right parent cond_ed str
                right=None,
            )
        self.mtd_bidep = None
        self.api_version = API_VERSION
        self.marginal_backend = "grid"
        self.marginal_backend_config = {}
        self.bicop_backend = DEFAULT_BICOP_BACKEND
        self.bicop_backend_config = {}
        self.tree_bidep = [{} for _ in range(num_dim - 1)]
        self.num_step_grid = num_step_grid
        self.sample_order = tuple(_ for _ in range(num_dim))
        self.first_tree_vertex = tuple()
        self.register_buffer("num_obs", torch.zeros((), dtype=torch.int))
        # ! device agnostic
        self.register_buffer("_dd", torch.tensor([], dtype=torch.float64))
        self._artifact: VineBuildArtifact | None = None
        object.__setattr__(self, "engine", VineCopEngine(artifact=None))

    @property
    def device(self):
        """
        Device of internal buffers.
        """
        return self._dd.device

    @property
    def dtype(self):
        """
        Data type of internal buffers.
        """
        return self._dd.dtype

    @property
    def matrix(self) -> torch.Tensor:
        """Matrix representation of the vine.

        Diagonal elements form the sampling order. Read in row-wise: a row of `(0, 1, 3, 4, 2)`
        indicates a source vertex `(0|1,2,3,4)` and bicops `(0,2;)`, `(0,4;2)`, `(0,3;2,4)`, and
        `(0,1;2,3,4)`.

        Returns:
            torch.Tensor: Matrix representation of the vine.
        """
        with torch.no_grad():
            mat: list[list[int]] = []
            seen: set[int] = set()
            # * iterate levels reversely; the diag‐variable is sample_order[idx]
            for idx, lv in enumerate(range(self.num_dim - 2, -1, -1)):
                v_diag = int(self.sample_order[idx])
                row: list[int] = [-1] * idx + [v_diag]
                # * look up tree-edges from level lv upwards
                for up in range(lv, -1, -1):
                    for v_l, v_r, *_ in self.tree_bidep[up]:
                        if v_diag in (v_l, v_r) and (v_l not in seen) and (v_r not in seen):
                            # * append the "other" variable
                            row.append(v_l if (v_diag == v_r) else v_r)
                seen.add(v_diag)
                mat.append(row)
            #
            mat.append([-1] * (self.num_dim - 1) + [int(self.sample_order[-1])])
            return torch.tensor(mat, dtype=torch.int, device=self.device)

    def log_pdf(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute the log-density function of the vine copula at the given observations.

        The input may include marginal-scale observations when the model was fitted with
        `is_cop_scale=False`.

        Args:
            obs (torch.Tensor): Points at which to evaluate the log-density function. Shape (num_obs, num_dim).

        Returns:
            torch.Tensor: Log-density function values at the given observations. Shape (num_obs, 1).
        """
        if self.engine.artifact is None:
            raise RuntimeError("log_pdf() requires a fitted model or installed execution plan.")
        return self.engine.log_pdf(obs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Neg average log-likelihood function for the given observations.

        Args:
            x (torch.Tensor): Points at which to evaluate. Shape (num_obs, num_dim).

        Returns:
            torch.Tensor: scalar loss value.
        """
        return -self.log_pdf(x).mean()

    def rosenblatt(self, obs: torch.Tensor, sample_order: tuple | None = None) -> torch.Tensor:
        """Compute the Rosenblatt transform of observations.

        Maps input `obs` into uniform pseudo‐observations via successive conditional CDFs (Rosenblatt).
        """
        if self.engine.artifact is not None:
            return self.engine.rosenblatt(obs=obs, sample_order=sample_order)
        raise RuntimeError("rosenblatt() requires a fitted model or installed execution plan.")

    @torch.no_grad()
    def inverse_rosenblatt(
        self,
        obs: torch.Tensor,
        sample_order: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if self.engine.artifact is not None:
            return self.engine.inverse_rosenblatt(obs=obs, sample_order=sample_order)
        raise RuntimeError(
            "inverse_rosenblatt() requires a fitted model or installed execution plan."
        )

    @torch.no_grad()
    def _sample_u(
        self,
        num_sample: int = 1000,
        seed: int | None = 42,
        is_sobol: bool = False,
        sample_order: tuple[int, ...] | None = None,
        dct_v_s_obs: dict[tuple[int, ...], torch.Tensor] | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if self.engine.artifact is None:
            raise RuntimeError("_sample_u() requires a fitted model or installed execution plan.")
        return self.engine._sample_u(
            num_sample=num_sample,
            seed=seed,
            is_sobol=is_sobol,
            sample_order=sample_order,
            dct_v_s_obs=dct_v_s_obs,
            generator=generator,
        )

    @torch.no_grad()
    def sample(
        self,
        num_sample: int = 1000,
        seed: int | None = 42,
        is_sobol: bool = False,
        sample_order: tuple[int, ...] | None = None,
        dct_v_s_obs: dict[tuple[int, ...], torch.Tensor] | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if self.engine.artifact is None:
            raise RuntimeError("sample() requires a fitted model or installed execution plan.")
        return self.engine.sample(
            num_sample=num_sample,
            seed=seed,
            is_sobol=is_sobol,
            sample_order=sample_order,
            dct_v_s_obs=dct_v_s_obs,
            generator=generator,
        )

    @torch.no_grad()
    def cdf(self, obs: torch.Tensor, num_sample: int = 10007, seed: int = 42) -> torch.Tensor:
        if self.engine.artifact is None:
            raise RuntimeError("cdf() requires a fitted model or installed execution plan.")
        return self.engine.cdf(obs=obs, num_sample=num_sample, seed=seed)

    def __str__(self) -> str:
        """String representation of the ``VineCop`` object.

        Returns:
            str: String representation of the ``VineCop`` object.
        """
        header = self.__class__.__name__
        params = {
            "num_dim": int(self.num_dim),
            "num_obs": int(self.num_obs),
            "is_cop_scale": self.is_cop_scale,
            "mtd_bidep": self.mtd_bidep,
            "marginal_backend": self.marginal_backend,
            "bicop_backend": self.bicop_backend,
            "boundary_policy": self.boundary_policy,
            "negloglik": float(
                sum(bcp.negloglik for bcp in self.bicops.values()).round(decimals=4)
            ),
            "num_step_grid": int(self.num_step_grid),
            "dtype": self.dtype,
            "device": self.device,
            "sample_order": self.sample_order,
        }
        params_str = pformat(params, sort_dicts=False, underscore_numbers=True)
        matrix_str = indent(str(self.matrix), " " * 4)
        return f"{header}\n{params_str[1:-1]},\n 'matrix':\n{matrix_str}\n\n"
