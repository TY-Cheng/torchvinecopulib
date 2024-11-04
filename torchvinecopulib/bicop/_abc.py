"""
Abstract bivariate copula from abstract base class (ABC):
* CamelCase naming convention for class
* data (par, rot as stored in DataBiCop) and functions (methods in children of BiCopAbstract) are decoupled
* sorted alphabetically for maintainability
* staticmethod by default, classmethod when reusing
    by default, calculate pdf_0 by taking cls.l_pdf_0().exp()
    where l_pdf_0() is staticmethod and pdf_0() is classmethod;
    override when doing it in reverse order.

! by default, we assume raw bicop observations (obs) is rotated COUNTER-CLOCKWISE, FROM an ideal un-rotated bicop obs:
! to reuse the function written for un-rotated bicop with suffix `_0`,
! as pretreatment, we do CLOCKWISE rotation FROM raw bicop obs to the ideal un-rotated bicop obs. (see rot_0 below)
e.g.

RAW BICOP:
A | B
  |
--d---
  |
C | D

IDEAL UN-ROTATED BICOP:
C'| A'
  |
--d'--
  |
D'| B'

Suppose a raw bicop obs is rotated 90 counter-clockwise from an ideal un-rotated bicop obs;
the `cdf` of raw bicop obs is the area (C), as segmented by the obs point (d) in the raw bicop;
which corresponds to the area (C') as segmented by the obs point (d') in the ideal un-rotated bicop;
suppose d=(v0,v1), then d'=(v1, 1-v0)
thus cdf(d)=C'+D'-D'=v1-cdf(d')
"""

from abc import ABC, abstractmethod

import torch

from ..util import _CDF_MAX, _CDF_MIN, _TAU_MAX, _TAU_MIN


class BiCopAbstract(ABC):
    _PAR_MIN, _PAR_MAX = tuple(), tuple()

    @classmethod
    def cdf(
        cls,
        obs: torch.Tensor,
        par: tuple,
        rot: int,
    ) -> torch.Tensor:
        col_p: torch.Tensor = cls.cdf_0(obs=cls.rot_0(obs=obs, rot=rot), par=par)
        if rot == 0:
            res = col_p
        elif rot == 90:
            res = obs[:, [1]] - col_p
        elif rot == 180:
            res = obs.sum(dim=1, keepdim=True) + col_p - 1.0
        elif rot == 270:
            res = obs[:, [0]] - col_p
        else:
            raise NotImplementedError
        return res.nan_to_num_().clamp_(min=_CDF_MIN, max=_CDF_MAX)

    @staticmethod
    @abstractmethod
    def cdf_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def hfunc1(
        cls,
        obs: torch.Tensor,
        par: tuple,
        rot: int,
    ) -> torch.Tensor:
        """first h function, Prob(V1<=v1 | V0=v0)"""
        if rot == 0:
            res = cls.hfunc1_0(obs=obs, par=par)
        elif rot == 90:
            res = cls.hfunc2_0(obs=cls.rot_0(obs=obs, rot=rot), par=par)
        elif rot == 180:
            res = 1.0 - cls.hfunc1_0(obs=cls.rot_0(obs=obs, rot=rot), par=par)
        elif rot == 270:
            res = 1.0 - cls.hfunc2_0(obs=cls.rot_0(obs=obs, rot=rot), par=par)
        else:
            raise NotImplementedError
        return res.nan_to_num_().clamp_(min=_CDF_MIN, max=_CDF_MAX)

    @staticmethod
    @abstractmethod
    def hfunc1_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def hfunc2(
        cls,
        obs: torch.Tensor,
        par: tuple,
        rot: int,
    ) -> torch.Tensor:
        """second h function, Prob(V0<=v0 | V1=v1)"""
        if rot == 0:
            res = cls.hfunc2_0(obs=obs, par=par)
        elif rot == 90:
            res = 1.0 - cls.hfunc1_0(obs=cls.rot_0(obs=obs, rot=rot), par=par)
        elif rot == 180:
            res = 1.0 - cls.hfunc2_0(obs=cls.rot_0(obs=obs, rot=rot), par=par)
        elif rot == 270:
            res = cls.hfunc1_0(obs=cls.rot_0(obs=obs, rot=rot), par=par)
        else:
            raise NotImplementedError
        return res.nan_to_num_().clamp_(min=_CDF_MIN, max=_CDF_MAX)

    @classmethod
    def hfunc2_0(cls, obs: torch.Tensor, par: tuple) -> torch.Tensor:
        return cls.hfunc1_0(obs=obs.fliplr(), par=par)

    @classmethod
    def hinv1(
        cls,
        obs: torch.Tensor,
        par: tuple,
        rot: int,
    ) -> torch.Tensor:
        """inverse of the first h function, Q(p=v1 | V0=v0)"""
        if rot == 0:
            res = cls.hinv1_0(obs=obs, par=par)
        elif rot == 90:
            res = cls.hinv2_0(obs=cls.rot_0(obs=obs, rot=rot), par=par)
        elif rot == 180:
            res = 1.0 - cls.hinv1_0(obs=cls.rot_0(obs=obs, rot=rot), par=par)
        elif rot == 270:
            res = 1.0 - cls.hinv2_0(obs=cls.rot_0(obs=obs, rot=rot), par=par)
        else:
            raise NotImplementedError
        return res.nan_to_num_().clamp_(min=_CDF_MIN, max=_CDF_MAX)

    @staticmethod
    @abstractmethod
    def hinv1_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def hinv2(
        cls,
        obs: torch.Tensor,
        par: tuple,
        rot: int,
    ) -> torch.Tensor:
        """inverse of the second h function, Q(p=v0 | V1=v1)"""
        if rot == 0:
            res = cls.hinv2_0(obs=obs, par=par)
        elif rot == 90:
            res = 1.0 - cls.hinv1_0(obs=cls.rot_0(obs=obs, rot=rot), par=par)
        elif rot == 180:
            res = 1.0 - cls.hinv2_0(obs=cls.rot_0(obs=obs, rot=rot), par=par)
        elif rot == 270:
            res = cls.hinv1_0(obs=cls.rot_0(obs=obs, rot=rot), par=par)
        else:
            raise NotImplementedError
        return res.nan_to_num_().clamp_(min=_CDF_MIN, max=_CDF_MAX)

    @classmethod
    def hinv2_0(cls, obs: torch.Tensor, par: tuple) -> torch.Tensor:
        return cls.hinv1_0(obs=obs.fliplr(), par=par)

    @classmethod
    def l_pdf(
        cls,
        obs: torch.Tensor,
        par: tuple,
        rot: int,
    ) -> torch.Tensor:
        """bicop log-density values"""
        return cls.l_pdf_0(obs=cls.rot_0(obs=obs, rot=rot), par=par).nan_to_num_()

    @staticmethod
    @abstractmethod
    def l_pdf_0(obs: torch.Tensor, par: tuple) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def negloglik(
        cls: "BiCopAbstract",
        obs: torch.Tensor,
        par: tuple,
        rot: int,
    ) -> float:
        return cls.l_pdf(obs=obs, par=par, rot=rot).sum().neg().item()

    @classmethod
    def par2tau(
        cls,
        par: tuple,
        rot: int,
    ) -> float:
        """Convert bicop parameters to Kendall's tau."""
        if rot in (0, 180):
            res = cls.par2tau_0(par=par)
        elif rot in (90, 270):
            res = -cls.par2tau_0(par=par)
        else:
            raise NotImplementedError
        return max(min(res, _TAU_MAX), _TAU_MIN)

    @staticmethod
    @abstractmethod
    def par2tau_0(par: tuple) -> float:
        raise NotImplementedError

    @classmethod
    def pdf(
        cls,
        obs: torch.Tensor,
        par: tuple,
        rot: int,
    ) -> torch.Tensor:
        """
        Calculate bicop density values from obs and parameters.
        """
        return (
            cls.pdf_0(obs=cls.rot_0(obs=obs, rot=rot), par=par)
            .nan_to_num_()
            .clamp_min_(min=_CDF_MIN)
        )

    @classmethod
    def pdf_0(cls, obs: torch.Tensor, par: tuple) -> torch.Tensor:
        return cls.l_pdf_0(obs=obs, par=par).exp()

    @staticmethod
    @torch.jit.script
    def rot_0(obs: torch.Tensor, rot: int) -> torch.Tensor:
        """
        Rotate bicop obs clockwise (inverse direction) back to '_0',
        to reuse bicop functions with suffix `_0`.
        """
        if rot == 0:
            return obs
        elif rot == 90:
            return torch.hstack([obs[:, [1]], 1.0 - obs[:, [0]]])
        elif rot == 180:
            return 1.0 - obs
        elif rot == 270:
            return torch.hstack([1.0 - obs[:, [1]], obs[:, [0]]])
        else:
            raise NotImplementedError

    @classmethod
    def sim(
        cls,
        par: tuple,
        num_sim: int,
        rot: int = 0,
        seed: int = 0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        # inverse Rosenblatt transform
        # * v1p2~Unif(IndepBiCop), hfunc1(v2|v1)=p2, hinv1(p2|v1)=v2
        torch.manual_seed(seed=seed)
        obs = torch.rand(size=(num_sim, 2), device=device, dtype=dtype)
        obs[:, [1]] = cls.hinv1(
            obs=obs,
            par=par,
            rot=rot,
        )
        return obs

    @classmethod
    def tau2par(cls, tau: float, **kwargs) -> tuple:
        """
        Convert Kendall's tau to bicop parameters.
        rotation ignored, for tau is symmetric.
        """
        return (
            (
                *(
                    max(min(val, cls._PAR_MAX[idx]), cls._PAR_MIN[idx])
                    for idx, val in enumerate(cls.tau2par_0(tau=tau, **kwargs))
                ),
            )
            if cls.__name__ != "Independent"
            else tuple()
        )

    @staticmethod
    @abstractmethod
    def tau2par_0(tau: float, **kwargs) -> tuple:
        raise NotImplementedError

    @classmethod
    def hinv1_num(cls, u: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        # TODO
        """First h inverse function using numerical inversion"""
        # Create a copy of the input matrix u
        u_new = u.clone()

        # Define the function h1
        def h1(v: torch.Tensor) -> torch.Tensor:
            u_new[:, [1]] = v  # Update the second column of u_new with v
            return cls.hfunc1_0(u_new, par)  # Call hfunc1_0 with the updated u_new

        # Numerically invert the function
        return cls.invert_f(u[:, [1]], h1)

    def invert_f(x: torch.Tensor, f):
        # TODO
        """
        Numerically invert a function using bisection method.
        Args:
            x (torch.Tensor): Input tensor of target values for which we want to find the inverse.
            f (callable): The function for which we want to compute the inverse.
        Returns:
            torch.Tensor: Inverted values (the x that satisfies f(x) = target).
        """
        lb = 1e-20
        ub = 1 - 1e-20
        n_iter = 35

        # Initialize bounds and temp variables
        xl = torch.full_like(x, lb)
        xh = torch.full_like(x, ub)
        x_tmp = x.clone()
        fm = torch.zeros_like(x)

        # Bisection method loop
        for _ in range(n_iter):
            x_tmp = (xh + xl) / 2.0  # Midpoint
            fm = f(x_tmp) - x  # Evaluate function and compare with target x

            # Update bounds based on the sign of fm
            xl = torch.where(fm < 0, x_tmp, xl)
            xh = torch.where(fm >= 0, x_tmp, xh)

        # Handle NaN values by replacing them with NaN
        if torch.isnan(fm).any():
            nan_mask = torch.isnan(fm)
            x_tmp[nan_mask] = float("nan")

        return x_tmp
