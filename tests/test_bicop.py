import logging
import unittest

import numpy as np
from pyvinecopulib import Bicop, FitControlsBicop

from torchvinecopulib.bicop import SET_FAMnROT, bcp_from_obs
from torchvinecopulib.util import _TAU_MAX, _TAU_MIN

from . import DCT_FAM, LST_MTD_FIT, compare_chart_vec, sim_from_bcp


def calc_fit_par(bcp_pvc, bcp_tvc, rot: int, mtd_fit: str | None = None) -> tuple | None:
    """
    given bcp, fam, rot, prepare vector of parameters to fit
    simulate based on fam/rot/par, then compare the fitted parameters given mtd_fit
    """
    if bcp_tvc.__name__ == "Independent":
        assert (
            bcp_from_obs(sim_from_bcp(bcp_tvc=bcp_tvc), tpl_fam=(bcp_tvc.__name__,)).par == tuple()
        )
        return None
    elif bcp_tvc.__name__ == "StudentT":
        vec_par = np.array(
            tuple(
                zip(
                    np.linspace(-0.83, 0.83, 13),
                    # * nu from both int and float
                    np.concatenate([np.arange(2, 8), np.linspace(2.1, 10, 7)]),
                )
            )
        )
    elif bcp_tvc.__name__ == "Bb1":
        vec_par = np.array(
            tuple(
                zip(
                    np.linspace(0.1, 6.99, 13),
                    np.linspace(1.01, 6.99, 13),
                )
            )
        )
    else:
        vec_par = np.linspace(
            (bcp_tvc._PAR_MIN[0] + bcp_tvc._PAR_MAX[0] * 0.02) / 2,
            bcp_tvc._PAR_MAX[0] / 5.5,
            13,
        )
    temp_fcb = FitControlsBicop(family_set=[bcp_pvc], parametric_method=mtd_fit)
    lst_pvc = []
    lst_tvc = []
    for i_par in vec_par:
        if bcp_tvc.__name__ == "StudentT":
            obs = sim_from_bcp(bcp_tvc=bcp_tvc, par=i_par, rot=rot, num_sim=2000)
            lst_pvc.append(np.abs(Bicop(data=obs.cpu(), controls=temp_fcb).parameters).sum())
            lst_tvc.append(
                np.abs(
                    bcp_from_obs(
                        obs_bcp=obs, thresh_trunc=1, mtd_fit=mtd_fit, tpl_fam=[bcp_tvc.__name__]
                    ).par
                ).sum()
            )
        elif bcp_tvc.__name__ == "Bb1" and mtd_fit == "itau":
            continue
        elif bcp_tvc.__name__ == "Bb1" and mtd_fit == "mle":
            obs = sim_from_bcp(bcp_tvc=bcp_tvc, par= i_par, rot=rot, num_sim=2000)
            lst_pvc.append(np.abs(Bicop(data=obs.cpu(), controls=temp_fcb).parameters).sum())
            lst_tvc.append(
                np.abs(
                    bcp_from_obs(
                        obs_bcp=obs, thresh_trunc=1, mtd_fit=mtd_fit, tpl_fam=[bcp_tvc.__name__]
                    ).par
               ).sum()
            )
        else:
            obs = sim_from_bcp(bcp_tvc=bcp_tvc, par=(i_par,), rot=rot, num_sim=2000)
            lst_pvc.append(Bicop(data=obs.cpu(), controls=temp_fcb).parameters.item())
            lst_tvc.append(
                bcp_from_obs(
                    obs_bcp=obs, thresh_trunc=1, mtd_fit=mtd_fit, tpl_fam=[bcp_tvc.__name__]
                ).par[0]
            )
    return np.array(lst_pvc), np.array(lst_tvc)


def calc_cdfhfunchinvlpdf(
    bcp_pvc,
    bcp_tvc,
    obs_bcp,
    func_name: str = None,
) -> tuple:
    """compare instance methods from fitted bcp_pvc and static methods from bcp_tvc"""
    par = tuple(bcp_pvc.parameters.flatten())
    rot = bcp_pvc.rotation
    if func_name == "l_pdf":
        vec_pvc = np.log(bcp_pvc.pdf(obs_bcp.cpu()))
        vec_tvc = bcp_tvc.l_pdf(obs=obs_bcp, par=par, rot=rot).flatten().cpu().numpy()
    elif func_name in ("cdf", "hfunc1", "hinv1", "hfunc2", "hinv2"):
        vec_pvc = getattr(bcp_pvc, func_name)(obs_bcp.cpu())
        vec_tvc = (
            getattr(bcp_tvc, func_name)(obs=obs_bcp, par=par, rot=rot).flatten().cpu().numpy()
        )
    else:
        raise ValueError("Unknown test type")
    return vec_pvc, vec_tvc


def calc_tau2par2tau(bcp_pvc, bcp_tvc, func_name: str = None, rot: int = 0) -> tuple:
    """two-pass test for 'tau2par' and 'par2tau'
    start from a range of tau2par for 'tau2par', append par2tau further for 'par2tau'
    'tau2par' skipped for StudentT and Independent
    """
    if func_name in ("tau2par", "par2tau"):
        # ! notice the range of tau
        vec_tau = np.linspace(_TAU_MIN / 1.2, _TAU_MAX / 1.2, 100)
        if bcp_tvc.__name__ in ("StudentT", "Independent"):
            vec_pvc = vec_tau
            vec_tvc = vec_tau
        else:
            vec_pvc = np.array(tuple(map(bcp_pvc.tau_to_parameters, vec_tau))).flatten()
            vec_tvc = np.array(
                tuple(map(lambda tau: bcp_tvc.tau2par(tau=tau, rot=rot)[0], vec_tau))
            )
        if func_name == "par2tau":
            vec_pvc = np.array(
                tuple(map(lambda par: bcp_pvc.parameters_to_tau(np.array([par])), vec_pvc))
            ).flatten()
            vec_tvc = np.array(
                tuple(map(lambda par: bcp_tvc.par2tau(par=(par,), rot=rot), vec_tvc))
            )
        return vec_pvc, vec_tvc
    else:
        raise ValueError("Unknown test type")


class TestBiCop(unittest.TestCase):
    def setUp(self):
        pass

    def test_fit(self):
        """test the fit results (parameter, rotation) given obs"""
        for fam, rot in SET_FAMnROT:
            bcp_pvc, bcp_tvc = DCT_FAM[fam]
            for mtd_fit in LST_MTD_FIT:
                logging.info(
                    msg=f"\nTesting:\t{fam} {rot}\nComparing:\t{bcp_tvc} {bcp_pvc} {mtd_fit}"
                )
                title = f"{fam}_{rot}_{mtd_fit}"

                res = calc_fit_par(
                    bcp_pvc=bcp_pvc,
                    bcp_tvc=bcp_tvc,
                    rot=rot,
                    mtd_fit=mtd_fit,
                )
                if res:
                    vec_pvc, vec_tvc = res
                    if err := compare_chart_vec(
                        vec_pvc=vec_pvc,
                        vec_tvc=vec_tvc,
                        rtol=0.01,
                        atol=1.0,
                        title=title,
                        label=mtd_fit,
                    ):
                        logging.error(msg=err)
                        raise err

    def test_bcp_cdfhfunc1hinv1lpdf(self):
        """test the BiCop methods (cdf, hfunc1, hinv1, l_pdf) given obs"""
        for fam, rot in SET_FAMnROT:
            bcp_pvc, bcp_tvc = DCT_FAM[fam]
            obs_bcp = sim_from_bcp(bcp_tvc=bcp_tvc, rot=rot)
            obs_bcp_clone = obs_bcp.clone()
            mtd_fit = LST_MTD_FIT[1]
            logging.info(msg=f"\nTesting:\t{fam} {rot}\nComparing:\t{bcp_tvc} {bcp_pvc} {mtd_fit}")

            temp_fcb = FitControlsBicop(family_set=[bcp_pvc], parametric_method=mtd_fit)
            data_pvc = Bicop(data=obs_bcp.cpu(), controls=temp_fcb)

            for func_name in [
                "cdf",
                "hfunc1",
                "hinv1",
                "hfunc2",
                "hinv2",
                "l_pdf",
            ]:
                logging.debug(msg=f"{func_name}")
                title = f"{fam}_{rot}_{mtd_fit}_{func_name}"
                vec_pvc, vec_tvc = calc_cdfhfunchinvlpdf(
                    bcp_pvc=data_pvc,
                    bcp_tvc=bcp_tvc,
                    obs_bcp=obs_bcp,
                    func_name=func_name,
                )
                assert (obs_bcp == obs_bcp_clone).all()
                if err := compare_chart_vec(
                    vec_pvc=vec_pvc,
                    vec_tvc=vec_tvc,
                    title=title,
                    label=func_name,
                ):
                    logging.error(msg=err)
                    raise err

    def test_bcp_tau2par2tau(self):
        """test the BiCop methods (tau2par, par2tau)"""
        for fam, rot in SET_FAMnROT:
            if fam == "Independent" or fam == "Bb1":
                continue
            bcp_pvc, bcp_tvc = DCT_FAM[fam]
            obs_bcp = sim_from_bcp(bcp_tvc=bcp_tvc, rot=rot)
            obs_bcp_clone = obs_bcp.clone()
            mtd_fit = LST_MTD_FIT[1]
            logging.info(msg=f"\nTesting:\t{fam} {rot}\nComparing:\t{bcp_tvc} {bcp_pvc} {mtd_fit}")
            temp_fcb = FitControlsBicop(family_set=[bcp_pvc], parametric_method=mtd_fit)
            data_pvc = Bicop(data=obs_bcp.cpu(), controls=temp_fcb)
            for func_name in ("tau2par", "par2tau"):
                logging.debug(msg=f"{func_name}")
                title = f"{fam}_{rot}_{mtd_fit}_{func_name}"
                vec_pvc, vec_tvc = calc_tau2par2tau(
                    bcp_pvc=data_pvc,
                    bcp_tvc=bcp_tvc,
                    func_name=func_name,
                    rot=rot,
                )
                assert (obs_bcp == obs_bcp_clone).all()
                if err := compare_chart_vec(
                    vec_pvc=vec_pvc,
                    vec_tvc=vec_tvc,
                    title=title,
                    label=func_name,
                ):
                    logging.error(msg=err)
                    raise err


if __name__ == "__main__":
    unittest.main()
