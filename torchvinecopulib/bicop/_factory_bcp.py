import numpy as np
import torch
from scipy.optimize import minimize

from ..util import kendall_tau
from ._data_bcp import ENUM_FAM_BICOP, DataBiCop, SET_FAMnROT


def bcp_from_obs(
    obs_bcp: torch.Tensor,
    tau: float | None = None,
    thresh_trunc: float = 0.05,
    mtd_fit: str = "itau",
    mtd_mle: str = "L-BFGS-B",
    mtd_sel: str = "aic",
    tpl_fam: tuple[str, ...] = (
        "Clayton",
        "Frank",
        "Gaussian",
        "Gumbel",
        "Independent",
        "Joe",
    ),
    topk: int = 2,
) -> DataBiCop:
    """factory method to make a bivariate copula dataclass object, fitted from observations

    :param obs_bcp: bivariate copula obs, of shape (num_obs, 2) with values in [0, 1]
    :type obs_bcp: torch.Tensor
    :param tau: Kendall's tau of the observations, defaults to None for the function to estimate
    :type tau: float, optional
    :param thresh_trunc: threshold of Kendall's tau independence test, below which we reject independent bicop,
        defaults to 0.05
    :type thresh_trunc: float, optional
    :param mtd_fit: parameter estimation method, either 'itau' (inverse of tau) or
        'mle' (maximum likelihood estimation); defaults to "itau"
    :type mtd_fit: str, optional
    :param mtd_mle: optimization method for mle as used by scipy.optimize.minimize, defaults to "COBYLA"
    :type mtd_mle: str, optional
    :param mtd_sel: model selection criterion, either 'aic' or 'bic'; defaults to "aic"
    :type mtd_sel: str, optional
    :param tpl_fam: tuple of str as candidate family names to fit,
        could be a subset of ('Clayton', 'Frank', 'Gaussian', 'Gumbel', 'Independent', 'Joe', 'StudentT')
    :type tpl_fam: tuple, optional
    :param topk: number of best itau fit taken into further mle, used when mtd_fit is 'mle'; defaults to 2
    :type topk: int, optional
    :raises NotImplementedError: when mtd_fit is neither 'itau' nor 'mle'
    :return: fitted bivariate copula dataclass object
    :rtype: DataBiCop
    """
    # things known before fitting
    num_obs = obs_bcp.shape[0]
    # * tau from data, for inv-tau/tau2par, whose par is taken as init value for mle
    if tau is None:
        tau, pval = kendall_tau(x=obs_bcp[:, [0]], y=obs_bcp[:, [1]])
        if pval >= thresh_trunc:
            return DataBiCop(fam="Independent", negloglik=0.0, num_obs=num_obs, par=tuple(), rot=0)

    def _fit_itau(i_fam: str, i_rot: int) -> DataBiCop:
        # fetch the class
        i_cls = ENUM_FAM_BICOP[i_fam].value
        # tau to par
        i_par = i_cls.tau2par(tau=tau, obs=obs_bcp)
        return DataBiCop(
            fam=i_fam,
            negloglik=i_cls.negloglik(obs=obs_bcp, par=i_par, rot=i_rot),
            num_obs=num_obs,
            par=i_par,
            rot=i_rot,
        )

    # * tuple of a family tuple and a rotation tuple (transpose SET_FAMROT_ALL)
    fam_rot = (tpl for tpl in SET_FAMnROT if tpl[0] in tpl_fam)
    vec_bcp_data = np.fromiter((_fit_itau(*tpl) for tpl in fam_rot), dtype=DataBiCop)
    # ! take note `k` best to accelerate MLE
    idx_sel = np.argsort(a=tuple(_.__getattribute__(mtd_sel) for _ in vec_bcp_data))[:topk]
    # ! take studentt in (as its `itau` nll usually not good)
    idx = [idx for idx, _ in enumerate(vec_bcp_data) if _.fam == "StudentT"]
    if idx:
        idx = idx[0]
        if idx not in idx_sel:
            idx_sel = np.append(idx_sel, idx)
    vec_bcp_data = vec_bcp_data[idx_sel]

    if mtd_fit == "itau":
        # inv-tau: 'select' the min aic or bic
        return vec_bcp_data[0]

    elif mtd_fit == "mle":

        def _fit_mle(i_idx: int, i_fam: str, i_rot: int) -> DataBiCop:
            # quit early if 'Independent'
            if i_fam == "Independent":
                return vec_bcp_data[i_idx]
            # fetch the class
            i_cls = ENUM_FAM_BICOP[i_fam].value

            res = minimize(
                fun=lambda par: i_cls.l_pdf(par=par, obs=obs_bcp, rot=i_rot).sum().neg().item(),
                x0=vec_bcp_data[i_idx].par,
                bounds=tuple(zip(i_cls._PAR_MIN, i_cls._PAR_MAX)),
                method=mtd_mle,
            )
            return DataBiCop(
                fam=i_fam,
                negloglik=float(res.fun),
                num_obs=num_obs,
                par=tuple(res.x),
                rot=i_rot,
            )

        # * tuple of index and also fam_rot
        fam_rot = ((i, d.fam, d.rot) for i, d in enumerate(vec_bcp_data))
        vec_bcp_data = np.fromiter((_fit_mle(*tpl) for tpl in fam_rot), dtype=DataBiCop)
        # mle: 'select' the min aic or bic
        idx_sel = np.argmin(a=tuple(_.__getattribute__(mtd_sel) for _ in vec_bcp_data))
        return vec_bcp_data[idx_sel]

    else:
        raise NotImplementedError
