import math
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pprint import pformat


import torch

from ._abc import BiCopAbstract
from ._clayton import Clayton
from ._frank import Frank
from ._gaussian import Gaussian
from ._gumbel import Gumbel
from ._independent import Independent
from ._joe import Joe
from ._studentt import StudentT


class ENUM_FAM_BICOP(Enum):
    """an Enum class to store bivariate copula family names and their corresponding class objects"""

    Clayton: BiCopAbstract = Clayton
    Frank: BiCopAbstract = Frank
    Gaussian: BiCopAbstract = Gaussian
    Gumbel: BiCopAbstract = Gumbel
    Independent: BiCopAbstract = Independent
    Joe: BiCopAbstract = Joe
    StudentT: BiCopAbstract = StudentT


_FAM_ALL: list[str] = ENUM_FAM_BICOP._member_names_
# rotatable family
_FAM_ROT: tuple[str] = ("Clayton", "Gumbel", "Joe")

# * a set of tuples, each tuple contains a bivariate copula family name (str) and a rotation angle (int)
# * counter-clockwise rotation
SET_FAMnROT: set[tuple[str, int]] = {
    (fam, rot) if fam in _FAM_ROT else (fam, 0) for fam in _FAM_ALL for rot in (0, 90, 180, 270)
}


@dataclass(slots=True, frozen=True, kw_only=True)
class DataBiCop(ABC):
    """Dataclass for a bivariate copula.

    The default BiCopData is an Independent BiCop.

    num_obs = 1 by default, to avoid nan for bic.
    Can modify num_obs during instantiation, for accurate bic calculation
    """

    fam: str = "Independent"
    """bivariate copula family name"""

    negloglik: float = 0.0
    """negative log likelihood, recorded during fitting using observations"""

    num_obs: int = 1
    """number of observations, recorded during fitting using observations"""

    par: tuple = tuple()
    """parameters of the bivariate copula"""

    rot: int = 0
    """(COUNTER-CLOCKWISE) rotation degree of the bivariate copula model, one of (0, 90, 180, 270)"""

    @property
    def num_par(self) -> int:
        """number of parameters

        :param self: an instance of the DataBiCop dataclass
        :return: length of the parameter tuple of the bivariate copula dataclass object
        :rtype: int
        """
        return len(self.par)

    @property
    def aic(self) -> float:
        """

        :param self: an instance of the DataBiCop dataclass
        :return: Akaike information criterion (AIC)
        :rtype: float
        """
        return 2.0 * (self.num_par + self.negloglik)

    @property
    def bic(self) -> float:
        """

        :param self: an instance of the DataBiCop dataclass
        :return: Bayesian information criterion (BIC)
        :rtype: float
        """
        return 2.0 * self.negloglik + self.num_par * math.log(self.num_obs)

    @property
    def tau(self) -> float:
        """

        :param self: an instance of the DataBiCop dataclass
        :return: Kendall's tau
        :rtype: float
        """
        return ENUM_FAM_BICOP[self.fam].value.par2tau(par=self.par, rot=self.rot)

    def __str__(self) -> str:
        return pformat(
            object={
                "fam": self.fam,
                "rot": self.rot,
                "tau": round(self.tau, 4),
                "par": (*map(lambda _: round(_, 4), self.par),),
                "num_obs": self.num_obs,
                "negloglik": round(self.negloglik, 4),
                "aic": round(self.aic, 4),
                "bic": round(self.bic, 4),
            },
            compact=True,
            sort_dicts=False,
            underscore_numbers=True,
        )

    def cdf(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """

        :param self: an instance of the DataBiCop dataclass
        :param obs: observation of the bivariate copula, of shape (num_obs, 2)
        :type obs: torch.Tensor
        :return: cumulative distribution function (CDF) of shape (num_obs, 1), given the observation
        :rtype: torch.Tensor
        """
        return ENUM_FAM_BICOP[self.fam].value.cdf(
            obs=obs,
            par=self.par,
            rot=self.rot,
        )

    def hfunc1(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """

        :param self: an instance of the DataBiCop dataclass
        :param obs: observation of the bivariate copula, of shape (num_obs, 2)
        :type obs: torch.Tensor
        :return: first h function, Prob(V_right<=v_right | V_left=v_left), of shape (num_obs, 1), given the observation
        :rtype: torch.Tensor
        """
        return ENUM_FAM_BICOP[self.fam].value.hfunc1(
            obs=obs,
            par=self.par,
            rot=self.rot,
        )

    def hfunc2(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """

        :param self: an instance of the DataBiCop dataclass
        :param obs: observation of the bivariate copula, of shape (num_obs, 2)
        :type obs: torch.Tensor
        :return: second h function, Prob(V_left<=v_left | V_right=v_right), of shape (num_obs, 1), given the observation
        :rtype: torch.Tensor
        """
        return ENUM_FAM_BICOP[self.fam].value.hfunc2(
            obs=obs,
            par=self.par,
            rot=self.rot,
        )

    def hinv1(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """

        :param self: an instance of the DataBiCop dataclass
        :param obs: observation of the bivariate copula, of shape (num_obs, 2)
        :type obs: torch.Tensor
        :return: first h inverse function, Q(V_right=v_right | V_left=v_left), of shape (num_obs, 1), given the observation
        :rtype: torch.Tensor
        """
        return ENUM_FAM_BICOP[self.fam].value.hinv1(
            obs=obs,
            par=self.par,
            rot=self.rot,
        )

    def hinv2(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """

        :param self: an instance of the DataBiCop dataclass
        :param obs: observation of the bivariate copula, of shape (num_obs, 2)
        :type obs: torch.Tensor
        :return: second h inverse function, Q(V_left=v_left | V_right=v_right), of shape (num_obs, 1), given the observation
        :rtype: torch.Tensor
        """
        return ENUM_FAM_BICOP[self.fam].value.hinv2(
            obs=obs,
            par=self.par,
            rot=self.rot,
        )

    def l_pdf(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """

        :param self: an instance of the DataBiCop dataclass
        :param obs: observation of the bivariate copula, of shape (num_obs, 2)
        :type obs: torch.Tensor
        :return: log probability density function (PDF) of shape (num_obs, 1), given the observation
        :rtype: torch.Tensor
        """
        return ENUM_FAM_BICOP[self.fam].value.l_pdf(
            obs=obs,
            par=self.par,
            rot=self.rot,
        )

    def sim(
        self,
        num_sim: int,
        seed: int = 0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """

        :param self: an instance of the DataBiCop dataclass
        :param num_sim: number of simulations
        :type num_sim: int
        :param seed: random seed for torch.manual_seed(), defaults to 0
        :type seed: int, optional
        :param device: device for torch.rand(), defaults to 'cpu'
        :type device: str, optional
        :param dtype: data type for torch.rand(), defaults to torch.float64
        :type dtype: torch.dtype, optional
        :return: simulated observation of the bivariate copula, of shape (num_sim, 2)
        :rtype: torch.Tensor
        """
        torch.manual_seed(seed=seed)
        obs = torch.rand(size=(num_sim, 2), device=device, dtype=dtype)
        if self.fam == "Independent":
            return obs
        else:
            obs[:, [1]] = ENUM_FAM_BICOP[self.fam].value.hinv1(
                obs=obs,
                par=self.par,
                rot=self.rot,
            )
            return obs
