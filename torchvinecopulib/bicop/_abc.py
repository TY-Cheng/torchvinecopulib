"""
Abstract bivariate copula from abstract base class (ABC):
* CamelCase naming convention for class
* data (par, rot as stored in DataBiCop) and functions (methods in children of BiCopAbstract) are decoupled
* sorted alphabetically for maintainability
* staticmethod by default, classmethod when reusing
    by default, pdf_0() is from cls.l_pdf_0().exp()
        where l_pdf_0() is staticmethod and pdf_0() is classmethod
        override when doing it in reverse order
    by default, hfunc2_0() and hinv2_0() reuse hfunc1_0() and hinv1_0()
        ! assuming exchangeability !

! by default, we assume raw bicop obs are already rotated COUNTER-CLOCKWISE, FROM an ideal un-rotated bicop obs
! to reuse the function written for un-rotated bicop with suffix `_0`,
! as pretreatment, we do CLOCKWISE rotation FROM raw bicop obs to the ideal un-rotated bicop obs via rot_0()
e.g. rot_0(obs, 90)

raw bicop obs:
A | B
  |
--d---
  |
C | D

ideal un-rotated bicop obs:
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

* reflection/radial symmetry: c(1-u,1-v)=c(u,v), C(u,v)=C(1-u,1-v)+u+v-1
    copula's invariance under a 180-degree rotation about the center of the unit square.
    cop is identical to its survival cop
* exchange/permutation symmetry (exchangeability): C(u,v)=C(v,u)

Mangold, B. (2017). New concepts of symmetry for copulas (No. 06/2017).
"""

from abc import ABC, abstractmethod

import torch

from ..util import _CDF_MAX, _CDF_MIN, _TAU_MAX, _TAU_MIN, solve_ITP_vectorize


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
        # ! by default, assuming exchangeability
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

    @classmethod
    def hinv1_0(cls, obs: torch.Tensor, par: tuple[float]) -> torch.Tensor:
        """inverse of the first h function, Q(p=v1 | V0=v0), via root-finding"""
        vec_v0, vec_p = obs[:, [0]], obs[:, [1]]
        return solve_ITP_vectorize(
            fun=lambda vec_v1: cls.hfunc1_0(
                obs=torch.hstack([vec_v0, vec_v1]),
                par=par,
            )
            - vec_p,
            x_a=torch.zeros_like(vec_p),
            x_b=torch.ones_like(vec_p),
        )

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
        # ! by default, assuming exchangeability
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
        return cls.l_pdf(obs=obs, par=par, rot=rot).nan_to_num_().sum().neg().item()

    @classmethod
    def par2tau(
        cls,
        par: tuple,
        rot: int,
    ) -> float:
        """convert bicop parameters to Kendall's tau."""
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
        # * inverse Rosenblatt transform
        # * (v0,p)~Unif (IndepBiCop), hfunc1(v1|v0)=p, hinv1(p|v0)=v1
        torch.manual_seed(seed=seed)
        obs = torch.rand(size=(num_sim, 2), device=device, dtype=dtype)
        obs[:, [1]] = cls.hinv1(
            obs=obs,
            par=par,
            rot=rot,
        )
        return obs

    @classmethod
    @abstractmethod
    def tau2par(cls, tau: float, **kwargs) -> tuple:
        raise NotImplementedError
