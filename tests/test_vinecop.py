import logging
import os
import random
import unittest

import networkx as nx
import pyvinecopulib as pvc

from torchvinecopulib.vinecop import vcp_from_json, vcp_from_obs, vcp_from_pth

from . import DCT_FAM, DEVICE, LST_MTD_FIT, LST_MTD_SEL, compare_chart_vec, sim_vcp_from_bcp


class TestVineCop(unittest.TestCase):
    def setUp(self):
        pass

    def test_first(self):
        """test vcp_from_obs given tpl_first when fitting cvine/dvine, check if they are prioritized"""
        mtd_fit = "itau"
        mtd_sel = "aic"
        num_dim = 10
        for len_first in (1, 3):
            for fam, (_, bcp_tvc) in DCT_FAM.items():
                V_mvcp = sim_vcp_from_bcp(bcp_tvc=bcp_tvc, num_sim=1000, num_dim=num_dim)
                for mtd_bidep in [
                    "kendall_tau",
                    "ferreira_tail_dep_coeff",
                    "chatterjee_xi",
                    "wasserstein_dist_ind",
                ]:
                    for mtd_vine in ("cvine", "dvine", "rvine"):
                        tpl_first = tuple(
                            {random.randint(0, num_dim - 1) for _ in range(len_first)}
                        )
                        len_first = len(tpl_first)
                        if fam in ("StudentT", "Independent") or mtd_bidep == "mutual_info":
                            continue
                        logging.info(
                            msg=f"\nTesting:\t{fam}\nComparing:\t{bcp_tvc}, {mtd_fit} {mtd_sel} {mtd_bidep}"
                        )
                        res_tvc = vcp_from_obs(
                            obs_mvcp=V_mvcp,
                            is_Dissmann=True,
                            mtd_vine=mtd_vine,
                            tpl_first=tpl_first,
                            mtd_bidep=mtd_bidep,
                            thresh_trunc=1,
                            mtd_fit=mtd_fit,
                            mtd_sel=mtd_sel,
                            tpl_fam=(fam, "Independent"),
                        )
                        assert set(res_tvc.tpl_sim[-len_first:]) == set(tpl_first)

    def test_sim_lpdf_pvc_cdf(self):
        """test the simulation, diagonal of structure matrix, l_pdf and cdf"""
        mtd_fit = "itau"
        mtd_bidep, tree_criterion = "kendall_tau", "tau"
        mtd_sel, selection_criterion = "aic", "aic"
        for fam, (bcp_pvc, bcp_tvc) in DCT_FAM.items():
            if fam in ("StudentT", "Independent"):
                continue
            logging.info(msg=f"\nTesting:\t{fam}\nComparing:\t{bcp_tvc} {bcp_pvc} {mtd_fit}")
            V_mvcp = sim_vcp_from_bcp(bcp_tvc=bcp_tvc, num_sim=1000)
            num_dim = V_mvcp.shape[1]
            # * struct: sim, cvine
            res_tvc = vcp_from_obs(
                obs_mvcp=V_mvcp,
                is_Dissmann=True,
                mtd_vine="cvine",
                tpl_first=[],
                matrix=None,
                mtd_bidep=mtd_bidep,
                thresh_trunc=1,
                mtd_fit=mtd_fit,
                mtd_sel=mtd_sel,
                tpl_fam=(fam, "Independent"),
            )
            res_sim = vcp_from_obs(
                obs_mvcp=res_tvc.sim(num_sim=20000, device=DEVICE),
                is_Dissmann=True,
                mtd_vine="cvine",
                tpl_first=[],
                matrix=None,
                mtd_bidep=mtd_bidep,
                thresh_trunc=1,
                mtd_fit=mtd_fit,
                mtd_sel=mtd_sel,
                tpl_fam=(fam, "Independent"),
            )
            assert sum([(a - b) for a, b in zip(res_tvc.tpl_sim, res_sim.tpl_sim)]) <= 1
            # * struct: sim, dvine
            res_tvc = vcp_from_obs(
                obs_mvcp=V_mvcp,
                is_Dissmann=True,
                mtd_vine="dvine",
                tpl_first=[],
                matrix=None,
                mtd_bidep=mtd_bidep,
                thresh_trunc=1,
                mtd_fit=mtd_fit,
                mtd_sel=mtd_sel,
                tpl_fam=(fam, "Independent"),
            )
            res_sim = vcp_from_obs(
                obs_mvcp=res_tvc.sim(num_sim=20000, device=DEVICE),
                is_Dissmann=True,
                mtd_vine="dvine",
                tpl_first=[],
                matrix=None,
                mtd_bidep=mtd_bidep,
                thresh_trunc=1,
                mtd_fit=mtd_fit,
                mtd_sel=mtd_sel,
                tpl_fam=(fam, "Independent"),
            )
            assert sum([(a - b) for a, b in zip(res_tvc.tpl_sim, res_sim.tpl_sim)]) <= 1
            # * l_pdf, rvine
            res_tvc = vcp_from_obs(
                obs_mvcp=V_mvcp,
                is_Dissmann=True,
                mtd_vine="rvine",
                tpl_first=[],
                matrix=None,
                mtd_bidep=mtd_bidep,
                thresh_trunc=1,
                mtd_fit=mtd_fit,
                mtd_sel=mtd_sel,
                tpl_fam=(fam, "Independent"),
            )
            assert abs(res_tvc.negloglik + res_tvc.l_pdf(V_mvcp).sum()) <= 1e-6
            # * struct: pvc, rvine
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
            diag_tvc = [res_tvc.matrix[i, i] for i in range(num_dim)]
            assert sum([(1 + a - b) for a, b in zip(diag_tvc, diag_pvc)]) <= 1
            # * cdf: pvc, rvine
            vec_tvc = res_tvc.cdf(obs_mvcp=V_mvcp, num_sim=30000).cpu().numpy().flatten()
            vec_pvc = res_pvc.cdf(V_mvcp.cpu(), N=20000, num_threads=4)
            if err := compare_chart_vec(
                vec_pvc=vec_pvc,
                vec_tvc=vec_tvc,
                atol=6e-2,
                title=f"vinecop_cdf_{fam}_{mtd_fit}",
                label="cdf",
            ):
                logging.error(msg=err)
                raise err
        return None

    def test_from_matrix(self):
        """test vcp_from_obs with tree indicated by matrix"""
        for fam, (_, bcp_tvc) in DCT_FAM.items():
            V_mvcp = sim_vcp_from_bcp(bcp_tvc=bcp_tvc, num_sim=1000)
            for mtd_fit in LST_MTD_FIT:
                for mtd_sel in LST_MTD_SEL:
                    for mtd_bidep in [
                        "kendall_tau",
                        "ferreira_tail_dep_coeff",
                        "chatterjee_xi",
                        "wasserstein_dist_ind",
                    ]:
                        if fam in ("StudentT", "Independent"):
                            continue
                        logging.info(
                            msg=f"\nTesting:\t{fam}\nComparing:\t{bcp_tvc}, {mtd_fit} {mtd_sel} {mtd_bidep}"
                        )
                        res_tvc = vcp_from_obs(
                            obs_mvcp=V_mvcp,
                            is_Dissmann=True,
                            matrix=None,
                            mtd_bidep=mtd_bidep,
                            thresh_trunc=1,
                            mtd_fit=mtd_fit,
                            mtd_sel=mtd_sel,
                            tpl_fam=(fam, "Independent"),
                        )
                        assert res_tvc == vcp_from_obs(
                            obs_mvcp=V_mvcp,
                            is_Dissmann=False,
                            matrix=res_tvc.matrix,
                            mtd_bidep=mtd_bidep,
                            thresh_trunc=1,
                            mtd_fit=mtd_fit,
                            mtd_sel=mtd_sel,
                            tpl_fam=(fam, "Independent"),
                        )

    def test_io(self):
        """test vcp_from_json, vcp_from_pth and vcp_to_json, vcp_to_pth"""
        fam = "Clayton"
        bcp_tvc = DCT_FAM[fam][1]
        mtd_fit = "itau"
        mtd_sel = "aic"
        num_dim = 6
        num_obs = 1000
        mtd_bidep = "kendall_tau"
        tpl_first = (3, 5)
        len_first = len(tpl_first)
        V_mvcp = sim_vcp_from_bcp(bcp_tvc=bcp_tvc, num_dim=num_dim, num_sim=num_obs)
        for mtd_vine in ("cvine", "dvine", "rvine"):
            logging.info(
                msg=f"\nTesting:\t\nComparing:\t{bcp_tvc}, {mtd_vine} {mtd_fit} {mtd_sel} {mtd_bidep}"
            )
            res_tvc = vcp_from_obs(
                obs_mvcp=V_mvcp,
                is_Dissmann=True,
                mtd_vine=mtd_vine,
                tpl_first=tpl_first,
                matrix=None,
                mtd_bidep=mtd_bidep,
                thresh_trunc=1,
                mtd_fit=mtd_fit,
                mtd_sel=mtd_sel,
            )
            # __str__
            dct_str = eval(str(res_tvc))
            assert dct_str["mtd_bidep"] == mtd_bidep
            assert dct_str["num_dim"] == num_dim
            assert dct_str["num_obs"] == num_obs
            if mtd_vine in ("cvine", "dvine"):
                assert set(dct_str["tpl_sim"][-len_first:]) == set(tpl_first)
        # draw
        path = "./vcp.png"
        fig, ax, G, tmp_p = res_tvc.draw_lv(f_path=path)
        assert isinstance(G, nx.Graph)
        os.remove(tmp_p)
        fig, ax, G, path = res_tvc.draw_dag(f_path=path)
        assert isinstance(G, nx.DiGraph)
        os.remove(path)
        # json
        path = "./vcp.json"
        tmp_p = res_tvc.vcp_to_json(f_path=path)
        tmp_f = vcp_from_json(tmp_p)
        assert tmp_f == res_tvc
        os.remove(tmp_p)
        # pth
        path = "./vcp.pth"
        tmp_p = res_tvc.vcp_to_pth(f_path=path)
        tmp_f = vcp_from_pth(tmp_p)
        assert tmp_f == res_tvc
        os.remove(tmp_p)


if __name__ == "__main__":
    unittest.main()
