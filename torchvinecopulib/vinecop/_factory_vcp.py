import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from ..bicop import ENUM_FAM_BICOP, bcp_from_obs
from ..util import ENUM_FUNC_BIDEP
from ._data_vcp import DataVineCop


def _mst_bicop(lst_key_obs: list, dct_edge_lv: dict) -> list:
    """Construct Kruskal's MAXIMUM spanning tree (MST) from bivariate copula edges, using modified disjoint set/ union find.

    :param lst_key_obs: list of keys of (pseudo) observations, each key is a tuple of (vertex, cond set)
    :type lst_key_obs: list
    :param dct_edge_lv: dictionary of edges (vertex_l, vertex_r, cond set) and corresponding bivariate dependence metric value
    :type dct_edge_lv: dict
    :return: list of edges (vertex_l, vertex_r, cond set) in the MST
    :rtype: _type_
    """
    # * edge2tree (Kruskal's MST, disjoint set/ union find)
    # ! modify 'parent' to let pseudo obs linked to previous bicop edge
    parent = {k_obs: frozenset((k_obs[0], *k_obs[1])) for k_obs in lst_key_obs}
    parent = {**parent, **{v: v for _, v in parent.items()}}
    rank = {k_obs: 0 for k_obs in parent}
    mst = []

    def find(v):
        if parent[v] != v:
            parent[v] = find(parent[v])
        return parent[v]

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        else:
            parent[root_b] = root_a
            if rank[root_b] == rank[root_a]:
                rank[root_a] += 1

    # * notice we sort edges by ABS(bidep) in DESCENDING order
    for (v_l, v_r, s_and), bidep_abs in sorted(
        dct_edge_lv.items(), key=lambda x: abs(x[1]), reverse=True
    ):
        if find((v_l, s_and)) != find((v_r, s_and)):
            mst.append((v_l, v_r, s_and))
            union((v_l, s_and), (v_r, s_and))
    return mst


def vcp_from_obs(
    obs_mvcp: torch.Tensor,
    is_Dissmann: bool = True,
    matrix: np.array = None,
    mtd_bidep: str = "kendall_tau",
    mtd_fit: str = "itau",
    mtd_mle: str = "COBYLA",
    mtd_sel: str = "aic",
    tpl_fam: tuple = tuple(
        (_ for _ in ENUM_FAM_BICOP._member_names_ if _ != "StudentT")
    ),
    topk: int = 2,
) -> DataVineCop:
    """Construct a vine copula model from multivariate observations, with structure prescribed by either Dissmann's (MST per level) method or a given matrix.

    :param obs_mvcp: multivariate observations, of shape (num_obs, num_dim)
    :type obs_mvcp: torch.Tensor
    :param is_Dissmann: whether to use Dissmann's method, defaults to True
    :type is_Dissmann: bool, optional
    :param matrix: matrix of vine copula structure, of shape (num_dim, num_dim), used when is_Dissmann is False, defaults to None
    :type matrix: np.array, optional
    :param mtd_bidep: method to calculate bivariate dependence, one of "kendall_tau", "mutual_info", "ferreira_tail_dep_coeff", "chatterjee_xi", "wasserstein_dist_ind"; defaults to "kendall_tau"
    :type mtd_bidep: str, optional
    :param mtd_fit: method to fit bivariate copula, either 'itau' (inverse of tau) or 'mle' (maximum likelihood estimation); defaults to "itau"
    :type mtd_fit: str, optional
    :param mtd_mle: optimization method for mle as used by scipy.optimize.minimize, defaults to "COBYLA"
    :type mtd_mle: str, optional
    :param mtd_sel: bivariate copula model selection criterion, either 'aic' or 'bic'; defaults to "aic"
    :type mtd_sel: str, optional
    :param tpl_fam: tuple of str as candidate family names to fit bicops, could be a subset of ('Clayton', 'Frank', 'Gaussian', 'Gumbel', 'Independent', 'Joe', 'StudentT')
    :type tpl_fam: tuple, optional
    :param topk: number of best itau fit taken into further mle, used when mtd_fit is 'mle'; defaults to 2
    :type topk: int, optional
    :return: a fitted DataVineCop object
    :rtype: DataVineCop
    """
    f_bidep = ENUM_FUNC_BIDEP[mtd_bidep]._value_
    num_dim = obs_mvcp.shape[1]
    r_D, r_D1 = range(num_dim), range(num_dim - 1)
    dct_obs = {_: {} for _ in r_D}
    dct_edge = {_: {} for _ in r_D1}
    # * tree is either from Dissmann/MST or from matrix
    dct_tree = {_: {} for _ in r_D1} if is_Dissmann else dct_edge
    dct_bcp = {_: {} for _ in r_D1}
    for lv in r_D1:
        # ! lv_0 obs, preprocess to append a cond frozenset (s_and)
        if lv == 0:
            dct_obs[0] = {(idx, frozenset()): obs_mvcp[:, [idx]] for idx in r_D}
        if is_Dissmann:
            # Dissmann, J., Brechmann, E. C., Czado, C., & Kurowicka, D. (2013).
            # Selecting and estimating regular vine copulae and application to financial returns. Computational Statistics & Data Analysis, 59, 52-69.
            # * obs2edge, list possible edges that connect two obs, calc f_bidep
            lst_key_obs = dct_obs[lv].keys()
            for tpl_l, tpl_r in combinations(lst_key_obs, 2):
                # ! only those obs with same 'cond set' (the frozen set) can have edges (proximity condition)
                if tpl_l[1] == tpl_r[1]:
                    # ! sorted!
                    lst_xor = sorted((tpl_l[0], tpl_r[0]))
                    s_and = tpl_l[1]
                    dct_edge[lv][(*lst_xor, s_and)] = f_bidep(
                        x=dct_obs[lv][lst_xor[0], s_and],
                        y=dct_obs[lv][lst_xor[1], s_and],
                    )

            # * edge2tree (Kruskal's MST, disjoint set/ union find)
            # ! modify 'parent' to link pseudo obs to previous bicop edge
            dct_tree[lv] = {
                key_edge_mst: dct_edge[lv][key_edge_mst]
                for key_edge_mst in _mst_bicop(
                    lst_key_obs=lst_key_obs, dct_edge_lv=dct_edge[lv]
                )
            }
        else:
            # * tree structure is inferred from matrix, if not Dissmann
            for idx in range(num_dim - lv - 1):
                # ! sorted!
                lst_xor = sorted((matrix[idx, idx], matrix[idx, num_dim - lv - 1]))
                s_and = frozenset(matrix[idx, (num_dim - lv) :])
                dct_tree[lv][(*lst_xor, s_and)] = f_bidep(
                    x=dct_obs[lv][lst_xor[0], s_and],
                    y=dct_obs[lv][lst_xor[1], s_and],
                )

        # * tree2bicop, fit bicop & prepare pseudo obs for next lv
        for (v_l, v_r, s_and), bidep in dct_tree[lv].items():
            V_bcp = torch.hstack(
                [dct_obs[lv][v_l, s_and], dct_obs[lv][v_r, s_and]],
            )
            i_bcp = bcp_from_obs(
                obs_bcp=V_bcp,
                tau=bidep if mtd_bidep == "kendall_tau" else None,
                mtd_fit=mtd_fit,
                mtd_mle=mtd_mle,
                mtd_sel=mtd_sel,
                tpl_fam=tpl_fam,
                topk=topk,
            )
            dct_bcp[lv][(v_l, v_r, s_and)] = i_bcp
            # ! notice hfunc1 or hfunc2
            i_lv_next = lv + 1
            dct_obs[i_lv_next][v_r, s_and | {v_l}] = i_bcp.hfunc1(V_bcp)
            dct_obs[i_lv_next][v_l, s_and | {v_r}] = i_bcp.hfunc2(V_bcp)

    return DataVineCop(dct_tree=dct_tree, dct_bcp=dct_bcp, mtd_bidep=mtd_bidep)


def vcp_from_json(f_path: Path = Path("./vcp.json")) -> DataVineCop:
    """load a DataVineCop from a json file

    :param f_path: path to the json file, defaults to Path("./vcp.json")
    :type f_path: Path, optional
    :return: a DataVineCop object
    :rtype: DataVineCop
    """
    from ..bicop import DataBiCop

    with open(f_path, "r") as file:
        obj = eval(file.read())
    return DataVineCop(
        dct_bcp=obj["dct_bcp"], dct_tree=obj["dct_tree"], mtd_bidep=obj["mtd_bidep"]
    )


def vcp_from_pkl(f_path: Path = Path("./vcp.pkl")) -> DataVineCop:
    """load a DataVineCop from a pickle file

    :param f_path: path to the pickle file, defaults to Path("./vcp.pkl")
    :type f_path: Path, optional
    :return: a DataVineCop object
    :rtype: DataVineCop
    """
    with open(f_path, "rb") as file:
        obj = pickle.load(file)
    return obj
