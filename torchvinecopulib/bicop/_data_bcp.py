from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pprint import pformat

import torch

from ._abc import BiCopAbstract
from ._bb1 import BB1
from ._bb5 import BB5
from ._bb6 import BB6
from ._bb7 import BB7
from ._bb8 import BB8
from ._clayton import Clayton
from ._frank import Frank
from ._gaussian import Gaussian
from ._gumbel import Gumbel
from ._independent import Independent
from ._joe import Joe
from ._studentt import StudentT
from ._tawn1 import Tawn1
from ._tawn2 import Tawn2


class ENUM_FAM_BICOP(Enum):
    """an Enum class to store bivariate copula family names and their corresponding class objects"""

    Clayton: BiCopAbstract = Clayton
    Frank: BiCopAbstract = Frank
    Gaussian: BiCopAbstract = Gaussian
    Gumbel: BiCopAbstract = Gumbel
    Independent: BiCopAbstract = Independent
    Joe: BiCopAbstract = Joe
    StudentT: BiCopAbstract = StudentT
    BB1: BiCopAbstract = BB1
    BB5: BiCopAbstract = BB5
    BB6: BiCopAbstract = BB6
    BB7: BiCopAbstract = BB7
    BB8: BiCopAbstract = BB8
    Tawn1: BiCopAbstract = Tawn1
    Tawn2: BiCopAbstract = Tawn2


_FAM_ALL: list[str] = list(ENUM_FAM_BICOP.__members__.keys())
# * rotatable family
_FAM_ROT: tuple[str] = (
    "Clayton",
    "Gumbel",
    "Joe",
    "BB1",
    "BB5",
    "BB6",
    "BB7",
    "BB8",
    "Tawn1",
    "Tawn2",
)
# * a set of tuples, each tuple contains a bivariate copula family name (str) and a rotation angle (int)
# * counter-clockwise rotation
SET_FAMnROT: set[tuple[str, int]] = {
    (fam, rot) if fam in _FAM_ROT else (fam, 0)
    for fam in _FAM_ALL
    for rot in (0, 90, 180, 270)
}
SET_FAM_ITAU: set[str] = {
    "Clayton",
    "Frank",
    "Gaussian",
    "Gumbel",
    "Independent",
    "Joe",
}
SET_FAM_ITAU_QMLE: set[str] = {
    "BB1",
    "StudentT",
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

    neg_log_lik: torch.Tensor = torch.tensor(
        0.0,
        device="cpu",
        dtype=torch.float64,
    )
    """negative log likelihood, recorded during fitting using observations"""

    num_obs: torch.Tensor = torch.tensor(
        1,
        device="cpu",
        dtype=torch.float64,
    )
    """number of observations, recorded during fitting using observations"""

    par: torch.Tensor = torch.empty(0, device="cpu", dtype=torch.float64)
    """parameters of the bivariate copula"""

    rot: int = 0
    """(COUNTER-CLOCKWISE) rotation degree of the bivariate copula model, one of (0, 90, 180, 270)"""

    @property
    def num_par(self) -> torch.Tensor:
        """number of parameters
        :param self: an instance of the DataBiCop dataclass
        :return: length of the parameter tuple of the bivariate copula dataclass object
        :rtype: torch.Tensor
        """
        return torch.tensor(
            self.par.numel(),
            device=self.par.device,
            dtype=self.par.dtype,
        )

    @property
    def aic(self) -> torch.Tensor:
        """
        :param self: an instance of the DataBiCop dataclass
        :return: Akaike information criterion (AIC)
        :rtype: torch.Tensor
        """
        return 2.0 * (self.num_par + self.neg_log_lik)

    @property
    def bic(self) -> torch.Tensor:
        """
        :param self: an instance of the DataBiCop dataclass
        :return: Bayesian information criterion (BIC)
        :rtype: torch.Tensor
        """
        return 2.0 * self.neg_log_lik + self.num_par * self.num_obs.log()

    @property
    def tau(self) -> torch.Tensor:
        """
        :param self: an instance of the DataBiCop dataclass
        :return: Kendall's tau
        :rtype: torch.Tensor
        """
        return ENUM_FAM_BICOP[self.fam].value.par2tau(par=self.par, rot=self.rot)

    def __str__(self) -> str:
        return pformat(
            object={
                "fam": self.fam,
                "rot": self.rot,
                "tau": self.tau.round(decimals=4),
                "par": self.par.round(decimals=4),
                "num_obs": self.num_obs,
                "negloglik": self.neg_log_lik.round(decimals=4),
                "aic": self.aic.round(decimals=4),
                "bic": self.bic.round(decimals=4),
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

    def hfunc_l(
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
        return ENUM_FAM_BICOP[self.fam].value.hfunc_l(
            obs=obs,
            par=self.par,
            rot=self.rot,
        )

    def hfunc_r(
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
        return ENUM_FAM_BICOP[self.fam].value.hfunc_r(
            obs=obs,
            par=self.par,
            rot=self.rot,
        )

    def hinv_l(
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
        return ENUM_FAM_BICOP[self.fam].value.hinv_l(
            obs=obs,
            par=self.par,
            rot=self.rot,
        )

    def hinv_r(
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
        return ENUM_FAM_BICOP[self.fam].value.hinv_r(
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
        is_sobol: bool = False,
    ) -> torch.Tensor:
        """Simulates random samples from a bivariate copula model.

        :param num_sim: Number of samples to simulate.
        :type num_sim: int
        :param seed: Random seed for reproducibility. Defaults to 0.
        :type seed: int, optional
        :param device: Device to perform the computation ('cpu' or 'cuda'). Defaults to "cpu".
        :type device: str, optional
        :param dtype: Data type of the generated samples. Defaults to torch.float64.
        :type dtype: torch.dtype, optional
        :param is_sobol: If True, uses Sobol sequence for quasi-random number generation. Defaults to False.
        :type is_sobol: bool, optional
        :return: A tensor of shape (num_sim, 2) containing the simulated samples.
        :rtype: torch.Tensor
        """
        if is_sobol:
            obs = (
                torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)
                .draw(n=num_sim, dtype=dtype)
                .to(device)
            )
        else:
            torch.manual_seed(seed=seed)
            obs = torch.rand(size=(num_sim, 2), device=device, dtype=dtype)
        if self.fam != "Independent":
            obs[:, [1]] = ENUM_FAM_BICOP[self.fam].value.hinv_l(
                obs=obs,
                par=self.par,
                rot=self.rot,
            )
        return obs
