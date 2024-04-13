import pickle
from collections import deque
from itertools import combinations, product
from operator import itemgetter
from pathlib import Path
from typing import Deque

import numpy as np
import torch

from ..bicop import bcp_from_obs
from ..util import ENUM_FUNC_BIDEP
from ._data_vcp import DataVineCop


def _mst_from_edge_rvine(lst_key_obs: list, dct_edge_lv: dict) -> list:
    # TODO suboptimal dynamic MST (cut property) with s_first as priority set
    # build one MST (connecting bcp vertices) for s_first first, then add the rest (by )
    """Construct Kruskal's MAXIMUM spanning tree (MST) from bivariate copula edges, using modified disjoint set/ union find.

    :param lst_key_obs: list of keys of (pseudo) observations, each key is a tuple of (vertex, cond set)
    :type lst_key_obs: list
    :param dct_edge_lv: dictionary of edges (vertex_l, vertex_r, cond set) and corresponding bivariate dependence metric value
    :type dct_edge_lv: dict
    :return: list of edges (vertex_l, vertex_r, cond set) in the MST
    :rtype: _type_
    """
    # * edge2tree, rvine (Kruskal's MST, disjoint set/ union find)
    # ! modify 'parent' to let a pseudo obs vertex linked to its previous bicop vertex
    parent = {v_s_cond: frozenset((v_s_cond[0], *v_s_cond[1])) for v_s_cond in lst_key_obs}
    # * and bicop vertices are linked to themselves
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
    for (v_l, v_r, s_cond), bidep_abs in sorted(
        dct_edge_lv.items(), key=lambda x: abs(x[1]), reverse=True
    ):
        if find((v_l, s_cond)) != find((v_r, s_cond)):
            mst.append((v_l, v_r, s_cond))
            union((v_l, s_cond), (v_r, s_cond))
    return mst


def _mst_from_edge_cvine(
    lst_key_obs: list, dct_edge_lv: dict, deq_sim: Deque, s_first: set
) -> tuple:
    """Construct Kruskal's MAXIMUM spanning tree (MST) from bivariate copula edges, restricted to cvine

    :param lst_key_obs: list of keys of (pseudo) observations, each key is a tuple of (vertex, cond set)
    :type lst_key_obs: list
    :param dct_edge_lv: dictionary of edges (vertex_l, vertex_r, cond set) and corresponding bivariate dependence metric value
    :type dct_edge_lv: dict
    :param s_first: set of vertices that are prioritized to be first in the simulation workflow
    :type s_first: set
    :param deq_sim: deque of vertices, to record the order of simulation (read from right to left, as simulated pseudo-obs vertices from shallowest to deepest level)
    :type deq_sim: Deque
    :return: list of edges (vertex_l, vertex_r, cond set) in the MST; updated deq_sim; updated s_first
    :rtype: tuple
    """
    # * edge2tree, cvine (MST, restricted to cvine)
    # init dict, the sum of abs bidep for each vertex
    if s_first:
        # * prioritize this set; only consider edges from/to vertices in this set
        dct_bidep_abs_sum = {v_s_cond: 0 for v_s_cond in lst_key_obs if v_s_cond[0] in s_first}
        dct_edge_lv = {
            (v_l, v_r, s_cond): bidep
            for ((v_l, v_r, s_cond), bidep) in dct_edge_lv.items()
            if ((v_l in s_first) or (v_r in s_first))
        }
    else:
        dct_bidep_abs_sum = {v_s_cond: 0 for v_s_cond in lst_key_obs}
    for (v_l, v_r, s_cond), bidep in dct_edge_lv.items():
        # cum sum of abs bidep for each vertex
        if (v_l, s_cond) in dct_bidep_abs_sum:
            dct_bidep_abs_sum[(v_l, s_cond)] += abs(bidep)
        if (v_r, s_cond) in dct_bidep_abs_sum:
            dct_bidep_abs_sum[(v_r, s_cond)] += abs(bidep)
    # center vertex (and its cond set) for cvine at this level
    v_c, s_cond_c = sorted(dct_bidep_abs_sum.items(), key=itemgetter(1))[-1][0]
    # record edges from/to the center vertex
    mst = [
        (v_l, v_r, s_cond)
        for (v_l, v_r, s_cond) in dct_edge_lv
        if ((s_cond == s_cond_c) and ((v_c == v_l) or (v_c == v_r)))
    ]
    # update the deq_sim, let those sim first be last in the deq_sim
    deq_sim.appendleft(v_c)
    if len(mst) == 1:
        # for the last lv, only one edge left, appendleft the other vertex into deq_sim
        deq_sim.appendleft(v_l if v_l != v_c else v_r)
    # update the priority set
    s_first -= {v_c}
    # * mst(cvine), deq_sim, s_first
    return mst, deq_sim, s_first


def _mst_from_edge_dvine(lst_key_obs: list, dct_edge_lv: dict, s_first: set) -> tuple:
    """Construct Kruskal's MAXIMUM spanning tree (MST) from bivariate copula edges, restricted to dvine

    :param lst_key_obs: list of keys of (pseudo) observations, each key is a tuple of (vertex, cond set)
    :type lst_key_obs: list
    :param dct_edge_lv: dictionary of edges (vertex_l, vertex_r, cond set) and corresponding bivariate dependence metric value
    :type dct_edge_lv: dict
    :param s_first: set of vertices that are prioritized to be first in the simulation workflow
    :type s_first: set
    :return: list of edges (vertex_l, vertex_r, cond set) in the MST (lv0); updated deq_sim; emptied s_first
    :rtype: tuple
    """
    # * edge2tree, dvine (MST, restricted to dvine)
    # * for dvine the whole struct (and sim flow) is known after lv0, and this func is only called at lv0.
    # ! at lv0, s_cond is known to be empty
    if len(lst_key_obs) < 3:
        # ! only two vertices
        v_head, v_tail = lst_key_obs
        v_head, v_tail = v_head[0], v_tail[0]
        mst = [(v_head, v_tail, frozenset())]
    else:
        import math
        from networkx import Graph
        from networkx.algorithms.approximation import threshold_accepting_tsp

        # ! more than two vertices
        dct_cost = {
            (v_l, v_r): math.log1p(1 / max(abs(bidep), 1e-10))
            for (v_l, v_r, s_cond), bidep in dct_edge_lv.items()
        }
        G = Graph()
        G.add_weighted_edges_from([(*k, v) for k, v in dct_cost.items()])
        s_rest = set(G.nodes) - s_first
        if (not s_first) or (not s_rest):
            # ! global TSP when one set is empty
            tsp = threshold_accepting_tsp(G, init_cycle="greedy")
            # * fetch the edges in the TSP, drop the one with max cost
            mst = [
                tuple(sorted((v, tsp[idx - 1]))) for idx, v in enumerate(tsp, start=0) if idx > 0
            ]
            s_l_r_cost = sorted(
                [(v_l, v_r, dct_cost[(v_l, v_r)]) for v_l, v_r in mst],
                key=lambda x: x[-1],
                reverse=True,
            )
            mst = s_l_r_cost[1:]
            mst = [(v_l, v_r, frozenset()) for (v_l, v_r, _) in mst]
            v_head, v_tail, _ = s_l_r_cost[0]
        elif len(s_first) in (1, len(lst_key_obs) - 1):
            # ! one set is a singleton
            tsp, v_head = (
                (s_first, list(s_rest)[0]) if len(s_rest) == 1 else (s_rest, list(s_first)[0])
            )
            # * TSP on the subgraph without the singleton
            tsp = threshold_accepting_tsp(G.subgraph(tsp), init_cycle="greedy")
            # * loop over the TSP and pick the combination with min cost
            # ! add head-neck, drop neck-tail
            tsp.append(tsp[1])
            cost = math.inf
            for idx, v in enumerate(tsp[1:-1], start=1):
                v_prev, v_next = tsp[idx - 1], tsp[idx + 1]
                # to drop
                if (cost_next := dct_cost[tuple(sorted((v, v_next)))]) > (
                    cost_prev := dct_cost[tuple(sorted((v, v_prev)))]
                ):
                    v_nebr, cost_nebr = v_next, cost_next
                else:
                    v_nebr, cost_nebr = v_prev, cost_prev
                # to add
                if (i_cost := dct_cost[tuple(sorted((v_head, v)))] - cost_nebr) < cost:
                    v_tail, v_neck, cost = v_nebr, v, i_cost
            mst = [(*sorted((v_head, v_neck)), frozenset())]
            for idx, v in enumerate(tsp[1:-1], start=1):
                v_l, v_r = sorted((v, tsp[idx + 1]))
                if {v_l, v_r} != {v_neck, v_tail}:
                    mst.append((v_l, v_r, frozenset()))
        else:
            # ! both sets have more than one vertex
            tsp_first = threshold_accepting_tsp(G.subgraph(s_first), init_cycle="greedy")
            tsp_first.append(tsp_first[1])
            tsp_rest = threshold_accepting_tsp(G.subgraph(s_rest), init_cycle="greedy")
            tsp_rest.append(tsp_rest[1])
            cost = math.inf
            # * loop over two TSPs and pick the combination with min cost
            for idx_neck, _v_neck in enumerate(tsp_first[1:-1], start=1):
                # * for the first TSP, drop head-neck
                v_prev, v_next = tsp_first[idx_neck - 1], tsp_first[idx_neck + 1]
                if (cost_next := dct_cost[tuple(sorted((_v_neck, v_next)))]) > (
                    cost_prev := dct_cost[tuple(sorted((_v_neck, v_prev)))]
                ):
                    v_nebr_neck, cost_nebr_neck = v_next, cost_next
                else:
                    v_nebr_neck, cost_nebr_neck = v_prev, cost_prev
                for idx_body, _v_body in enumerate(tsp_rest[1:-1], start=1):
                    # * for the second TSP, drop body-tail
                    v_prev, v_next = tsp_rest[idx_body - 1], tsp_rest[idx_body + 1]
                    if (cost_next := dct_cost[tuple(sorted((_v_body, v_next)))]) > (
                        cost_prev := dct_cost[tuple(sorted((_v_body, v_prev)))]
                    ):
                        v_nebr_body, cost_nebr_body = v_next, cost_next
                    else:
                        v_nebr_body, cost_nebr_body = v_prev, cost_prev
                    if (
                        i_cost := dct_cost[tuple(sorted((_v_neck, _v_body)))]
                        - cost_nebr_neck
                        - cost_nebr_body
                    ) < cost:
                        # ! cost, drop head-neck, drop body-tail, add neck-body
                        v_head, v_neck, v_body, v_tail, cost = (
                            v_nebr_neck,
                            _v_neck,
                            _v_body,
                            v_nebr_body,
                            i_cost,
                        )
            mst = [(*sorted((v_neck, v_body)), frozenset())]
            if len(tsp_first) == 4:
                mst.append((v_head, v_neck, frozenset()))
            else:
                for idx, v in enumerate(tsp_first[1:-1], start=1):
                    v_l, v_r = sorted((v, tsp_first[idx + 1]))
                    if {v_l, v_r} != {v_head, v_neck}:
                        mst.append((v_l, v_r, frozenset()))
            if len(tsp_rest) == 4:
                mst.append((v_body, v_tail, frozenset()))
            else:
                for idx, v in enumerate(tsp_rest[1:-1], start=1):
                    v_l, v_r = sorted((v, tsp_rest[idx + 1]))
                    if {v_l, v_r} != {v_body, v_tail}:
                        mst.append((v_l, v_r, frozenset()))
    # * for dvine, the whole struct (and deq_sim) is known after lv0
    deq_sim = deque([v_head])
    s_edge = set(mst)
    while deq_sim[-1] != v_tail:
        for v_l, v_r, s_cond in set(s_edge):
            if v_l == deq_sim[-1]:
                deq_sim.append(v_r)
                s_edge.remove((v_l, v_r, s_cond))
            elif v_r == deq_sim[-1]:
                deq_sim.append(v_l)
                s_edge.remove((v_l, v_r, s_cond))
    # ! let those sim first (s_first) be last in the lst_sim
    if deq_sim[0] in s_first:
        deq_sim.reverse()
    # * mst(dvine), deq_sim, s_first
    return mst, deq_sim, {}


def vcp_from_obs(
    obs_mvcp: torch.Tensor,
    is_Dissmann: bool = True,
    cdrvine: str = "rvine",
    lst_first: list[int] = [],
    matrix: np.ndarray | None = None,
    mtd_bidep: str = "chatterjee_xi",
    mtd_fit: str = "itau",
    mtd_mle: str = "COBYLA",
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
) -> DataVineCop:
    """Construct a vine copula model from multivariate observations, with structure prescribed by either Dissmann's (MST per level) method or a given matrix. May prioritize some vertices to be first in the cond simulation workflow.

    :param obs_mvcp: multivariate observations, of shape (num_obs, num_dim)
    :type obs_mvcp: torch.Tensor
    :param is_Dissmann: whether to use Dissmann's method or follow a given matrix, defaults to True; Dissmann, J., Brechmann, E. C., Czado, C., & Kurowicka, D. (2013). Selecting and estimating regular vine copulae and application to financial returns. Computational Statistics & Data Analysis, 59, 52-69.
    :type is_Dissmann: bool, optional
    :param cdrvine: one of 'cvine', 'dvine', 'rvine', defaults to "rvine"
    :type cdrvine: str, optional
    :param lst_first: list of vertices to be prioritized in the cond sim workflow, only used when cdrvine is "cvine" or "dvine". if empty then no priority is set, defaults to []
    :type lst_first: list[int], optional
    :param matrix: matrix of vine copula structure, of shape (num_dim, num_dim), used when is_Dissmann is False, defaults to None
    :type matrix: np.ndarray | None, optional
    :param mtd_bidep: method to calculate bivariate dependence, one of "kendall_tau", "mutual_info", "ferreira_tail_dep_coeff", "chatterjee_xi", "wasserstein_dist_ind", defaults to "chatterjee_xi"
    :type mtd_bidep: str, optional
    :param mtd_fit: method to fit bivariate copula, either 'itau' (inverse of tau) or 'mle' (maximum likelihood estimation); defaults to "itau"
    :type mtd_fit: str, optional
    :param mtd_mle: optimization method for mle as used by scipy.optimize.minimize, defaults to "COBYLA"
    :type mtd_mle: str, optional
    :param mtd_sel: bivariate copula model selection criterion, either 'aic' or 'bic'; defaults to "aic"
    :type mtd_sel: str, optional
    :param tpl_fam: tuple of str as candidate family names to fit bicop, could be a subset of ('Clayton', 'Frank', 'Gaussian', 'Gumbel', 'Independent', 'Joe', 'StudentT')
    :type tpl_fam: tuple[str, ...], optional
    :param topk: number of best itau fit taken into further mle, used when mtd_fit is 'mle'; defaults to 2
    :type topk: int, optional
    :raises ValueError: when cdrvine is not one of 'cvine', 'dvine', 'rvine'
    :return: a fitted DataVineCop object
    :rtype: DataVineCop
    """
    f_bidep = ENUM_FUNC_BIDEP[mtd_bidep]._value_
    num_dim = obs_mvcp.shape[1]
    s_first = set(lst_first)
    s_rest = set(range(num_dim)) - s_first
    # * an object to record the order of sim (read from right to left, as simulated pseudo-obs vertices from shallowest to deepest level)
    deq_sim = deque()
    r_D1 = range(num_dim - 1)
    dct_obs = {_: {} for _ in r_D1}
    # * tree is either from Dissmann (MST on edges, needs to record edges) or from matrix
    if is_Dissmann:
        dct_edge = {_: {} for _ in r_D1}
    dct_tree = {_: {} for _ in r_D1}
    dct_bcp = {_: {} for _ in r_D1}

    def _update_obs(v: int, s_cond: frozenset) -> torch.Tensor:
        """calc hfunc for pseudo obs and update dct_obs, only when necessary (lazy hfunc)"""
        # * v and s_cond are from the pseudo obs
        lv = len(s_cond)
        # * lv_1 and s_cond_1 mark the prev lv and bcp cond set
        lv_1 = lv - 1
        for (v_l, v_r, s_cond_1), bcp in dct_bcp[lv_1].items():
            # ! notice hfunc1 or hfunc2
            if (v == v_l) and (s_cond == frozenset({v_r} | s_cond_1)):
                dct_obs[lv][(v_l, s_cond)] = bcp.hfunc2(
                    obs=torch.hstack(
                        [
                            dct_obs[lv_1][v_l, s_cond_1],
                            dct_obs[lv_1][v_r, s_cond_1],
                        ]
                    )
                )
            elif (v == v_r) and (s_cond == frozenset({v_l} | s_cond_1)):
                dct_obs[lv][(v_r, s_cond)] = bcp.hfunc1(
                    obs=torch.hstack(
                        [
                            dct_obs[lv_1][v_l, s_cond_1],
                            dct_obs[lv_1][v_r, s_cond_1],
                        ]
                    )
                )

    for lv in r_D1:
        # ! lv_0 obs, preprocess to append an empty frozenset (s_cond)
        if lv == 0:
            dct_obs[0] = {(idx, frozenset()): obs_mvcp[:, [idx]] for idx in range(num_dim)}
        if is_Dissmann:
            # * obs2edge, list possible edges that connect two pseudo obs, calc f_bidep
            lst_key_obs = dct_obs[lv].keys()
            for (v_l, s_cond_l), (v_r, s_cond_r) in combinations(lst_key_obs, 2):
                # ! proximity condition: only those obs with same 'cond set' (the frozen set) can have edges
                if s_cond_l == s_cond_r:
                    # ! sorted !
                    v_l, v_r = sorted((v_l, v_r))
                    if dct_obs[lv][v_l, s_cond_l] is None:
                        _update_obs(v_l, s_cond_l)
                    if dct_obs[lv][v_r, s_cond_l] is None:
                        _update_obs(v_r, s_cond_l)
                    dct_edge[lv][(v_l, v_r, s_cond_l)] = f_bidep(
                        x=dct_obs[lv][v_l, s_cond_l],
                        y=dct_obs[lv][v_r, s_cond_l],
                    )
            if cdrvine == "dvine":
                # * edge2tree, dvine
                if lv == 0:
                    # ! for dvine the whole struct (and deq_sim) is known after lv0
                    mst, deq_sim, s_first = _mst_from_edge_dvine(
                        lst_key_obs=lst_key_obs,
                        dct_edge_lv=dct_edge[lv],
                        s_first=s_first,
                    )
                    dct_tree[lv] = {key_edge: dct_edge[lv][key_edge] for key_edge in mst}
                else:
                    dct_tree[lv] = dct_edge[lv]
            elif cdrvine == "cvine":
                # * edge2tree, cvine
                # ! for cvine, at each lv the center vertex is chosen to be the one with the largest sum of abs bidep
                mst, deq_sim, s_first = _mst_from_edge_cvine(
                    lst_key_obs=lst_key_obs,
                    dct_edge_lv=dct_edge[lv],
                    s_first=s_first,
                    deq_sim=deq_sim,
                )
                dct_tree[lv] = {key_edge: dct_edge[lv][key_edge] for key_edge in mst}
            elif cdrvine == "rvine":
                # * edge2tree, rvine
                mst = _mst_from_edge_rvine(lst_key_obs=lst_key_obs, dct_edge_lv=dct_edge[lv])
                dct_tree[lv] = {key_edge: dct_edge[lv][key_edge] for key_edge in mst}
            else:
                raise ValueError("cdrvine must be one of 'cvine', 'dvine', 'rvine'")
        else:
            # * tree structure is inferred from matrix, if not Dissmann
            for idx in range(num_dim - lv - 1):
                # ! sorted !
                v_l, v_r = sorted((matrix[idx, idx], matrix[idx, num_dim - lv - 1]))
                s_cond = frozenset(matrix[idx, (num_dim - lv) :])
                if dct_obs[lv][v_l, s_cond] is None:
                    _update_obs(v=v_l, s_cond=s_cond)
                if dct_obs[lv][v_r, s_cond] is None:
                    _update_obs(v=v_r, s_cond=s_cond)
                dct_tree[lv][(v_l, v_r, s_cond)] = f_bidep(
                    x=dct_obs[lv][(v_l, s_cond)],
                    y=dct_obs[lv][(v_r, s_cond)],
                )
        # * tree2bicop, fit bicop & record key of potential pseudo obs for next lv (lazy hfunc later)
        for (v_l, v_r, s_cond), bidep in dct_tree[lv].items():
            dct_bcp[lv][(v_l, v_r, s_cond)] = bcp_from_obs(
                obs_bcp=torch.hstack(
                    [dct_obs[lv][v_l, s_cond], dct_obs[lv][v_r, s_cond]],
                ),
                tau=bidep if mtd_bidep == "kendall_tau" else None,
                mtd_fit=mtd_fit,
                mtd_mle=mtd_mle,
                mtd_sel=mtd_sel,
                tpl_fam=tpl_fam,
                topk=topk,
            )
            i_lv_next = lv + 1
            # TODO: if s_rest is not empty, can filter out some pseudo-obs
            if i_lv_next < num_dim - 1:
                dct_obs[i_lv_next][v_r, s_cond | {v_l}] = None
                dct_obs[i_lv_next][v_l, s_cond | {v_r}] = None
        # ! garbage collection
        if is_Dissmann:
            del dct_edge[lv]
        if lv > 0:
            del dct_obs[lv - 1]
    # * for cvine/dvine, deq_sim is known and s_first is empty now
    # * for rvine, deq_sim is empty and to be filled by diag of matrix
    if not deq_sim:
        for lv in sorted(dct_tree, reverse=True):
            for v_l, v_r, s_cond in dct_tree[lv]:
                if (v_l not in deq_sim) and (v_r not in deq_sim):
                    # ! pick the node with smaller index (v_l < v_r), then mat <-> structure is bijection
                    deq_sim.append(v_l)
                    if lv == 0:
                        deq_sim.append(v_r)
                    break

    return DataVineCop(
        dct_bcp=dct_bcp,
        dct_tree=dct_tree,
        lst_sim=list(deq_sim),
        mtd_bidep=mtd_bidep,
    )


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
        dct_bcp=obj["dct_bcp"],
        dct_tree=obj["dct_tree"],
        lst_sim=obj["lst_sim"],
        mtd_bidep=obj["mtd_bidep"],
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
