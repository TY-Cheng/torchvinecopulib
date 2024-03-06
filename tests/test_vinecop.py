import logging
import os
import unittest

import pyvinecopulib as pvc
from torchvinecopulib.vinecop import vcp_from_json, vcp_from_obs, vcp_from_pkl

from . import (
    DCT_FAM,
    LST_MTD_BIDEP,
    LST_MTD_FIT,
    LST_MTD_SEL,
    compare_chart_vec,
    sim_vcp_from_bcp,
)


class TestVineCop(unittest.TestCase):
    def setUp(self):
        pass

    def test_sim_diag_lpdf_cdf(self):
        """test the simulation, diagonal of structure matrix, l_pdf and cdf"""
        mtd_fit = "itau"
        mtd_bidep, tree_criterion = "kendall_tau", "tau"
        mtd_sel, selection_criterion = "aic", "aic"
        for fam, (bcp_pvc, bcp_tvc) in DCT_FAM.items():
            if fam in ("StudentT", "Independent"):
                continue
            logging.info(
                msg=f"\nTesting:\t{fam}\nComparing:\t{bcp_tvc} {bcp_pvc} {mtd_fit}"
            )
            V_mvcp = sim_vcp_from_bcp(bcp_tvc=bcp_tvc)
            num_dim = V_mvcp.shape[1]
            res_tvc = vcp_from_obs(
                obs_mvcp=V_mvcp,
                is_Dissmann=True,
                matrix=None,
                mtd_bidep=mtd_bidep,
                mtd_fit=mtd_fit,
                mtd_sel=mtd_sel,
                tpl_fam=(fam, "Independent"),
            )
            res_sim = vcp_from_obs(
                obs_mvcp=res_tvc.sim(num_sim=20000),
                is_Dissmann=True,
                matrix=None,
                mtd_bidep=mtd_bidep,
                mtd_fit=mtd_fit,
                mtd_sel=mtd_sel,
                tpl_fam=(fam,),
            )
            # sim
            assert sum([(a - b) for a, b in zip(res_tvc.diag, res_sim.diag)]) <= 2
            res_pvc = pvc.Vinecop(
                data=V_mvcp.cpu(),
                controls=pvc.FitControlsVinecop(
                    family_set=[bcp_pvc, pvc.BicopFamily.indep],
                    parametric_method=mtd_fit,
                    tree_criterion=tree_criterion,
                    selection_criterion=selection_criterion,
                ),
            )
            # ! notice the order of the diagonal elements, and indexing starts from 1
            diag_pvc = [res_pvc.matrix[num_dim - 1 - i, i] for i in range(num_dim)]
            # diag
            assert sum([(1 + a - b) for a, b in zip(res_tvc.diag, diag_pvc)]) <= 2
            # l_pdf
            assert res_tvc.negloglik + res_tvc.l_pdf(V_mvcp).sum() <= 1e-6
            # cdf
            vec_tvc = (
                res_tvc.cdf(obs_mvcp=V_mvcp, num_sim=11000).cpu().numpy().flatten()
            )
            vec_pvc = res_pvc.cdf(V_mvcp.cpu(), N=11000)
            if err := compare_chart_vec(
                vec_pvc=vec_pvc,
                vec_tvc=vec_tvc,
                atol=2e-2,
                title=f"vinecop_cdf_{fam}_{mtd_fit}",
                label="cdf",
            ):
                logging.error(msg=err)
                raise err
        return None

    def test_from_matrix(self):
        """test vcp_from_obs with tree indicated by matrix"""
        for mtd_fit in LST_MTD_FIT:
            for mtd_sel in LST_MTD_SEL:
                for mtd_bidep in LST_MTD_BIDEP:
                    for fam, (_, bcp_tvc) in DCT_FAM.items():
                        if (
                            fam in ("StudentT", "Independent")
                            or mtd_bidep == "mutual_info"
                        ):
                            continue
                        logging.info(
                            msg=f"\nTesting:\t{fam}\nComparing:\t{bcp_tvc}, {mtd_fit} {mtd_sel} {mtd_bidep}"
                        )
                        V_mvcp = sim_vcp_from_bcp(bcp_tvc=bcp_tvc)
                        res_tvc = vcp_from_obs(
                            obs_mvcp=V_mvcp,
                            is_Dissmann=True,
                            matrix=None,
                            mtd_bidep=mtd_bidep,
                            mtd_fit=mtd_fit,
                            mtd_sel=mtd_sel,
                            tpl_fam=(fam, "Independent"),
                        )
                        assert res_tvc == vcp_from_obs(
                            obs_mvcp=V_mvcp,
                            is_Dissmann=False,
                            matrix=res_tvc.matrix,
                            mtd_bidep=mtd_bidep,
                            mtd_fit=mtd_fit,
                            mtd_sel=mtd_sel,
                            tpl_fam=(fam, "Independent"),
                        )

    def test_io(self):
        """test vcp_from_json, vcp_from_pkl and to_json, to_pkl"""
        fam = "Clayton"
        bcp_tvc = DCT_FAM[fam][1]
        mtd_fit = "itau"
        mtd_sel = "aic"
        mtd_bidep = "kendall_tau"
        logging.info(
            msg=f"\nTesting:\t\nComparing:\t{bcp_tvc}, {mtd_fit} {mtd_sel} {mtd_bidep}"
        )
        V_mvcp = sim_vcp_from_bcp(bcp_tvc=bcp_tvc)
        res_tvc = vcp_from_obs(
            obs_mvcp=V_mvcp,
            is_Dissmann=True,
            matrix=None,
            mtd_bidep=mtd_bidep,
            mtd_fit=mtd_fit,
            mtd_sel=mtd_sel,
        )
        tmp_p = res_tvc.to_json()
        __ = vcp_from_json(tmp_p)
        os.remove(tmp_p)
        assert __ == res_tvc

        tmp_p = res_tvc.to_pkl()
        __ = vcp_from_pkl(tmp_p)
        os.remove(tmp_p)
        assert __ == res_tvc


if __name__ == "__main__":
    unittest.main()
