import json
from ast import literal_eval
from collections import defaultdict, deque
from itertools import combinations
from operator import itemgetter
from pathlib import Path
from typing import Deque

import numpy as np
import torch

from ..bicop import DataBiCop, bcp_from_obs
from ..util import ENUM_FUNC_BIDEP, ref_count_hfunc
from ._data_vcp import DataVineCop


def _mst_from_edge_rvine(
    tpl_key_obs: tuple, dct_edge_lv: dict, s_first: set, lv: int, num_dim: int
) -> list:
    """Construct Kruskal's MAXIMUM spanning tree (MST) from bivariate copula edges,
    restricted to rvine, using modified disjoint set/ union find.

    :param tpl_key_obs: tuple of keys of (pseudo) observations, each key is a tuple of (vertex, cond set)
    :type tpl_key_obs: tuple
    :param dct_edge_lv: dictionary of edges (vertex_left, vertex_right, cond_set)
        and corresponding bivariate dependence metric value
    :type dct_edge_lv: dict
    :param s_first: set of vertices that are kept shallower in the simulation workflow (dynamically updated at each level)
    :type s_first: set
    :param lv: level, starting from 0
    :type lv: int
    :param num_dim: number of dimensions D
    :type num_dim: int
    :return: list of edges (vertex_left, vertex_right, cond set) in the MST
    :rtype: list
    """
    # * edge2tree, rvine (Kruskal's MST, disjoint set/ union find)
    # ! modify 'parent' to let a pseudo obs vertex linked to its previous bicop vertex
    parent = {v_s_cond: frozenset((v_s_cond[0], *v_s_cond[1])) for v_s_cond in tpl_key_obs}
    # * and bicop vertices are linked to themselves
    parent = {**parent, **{v: v for _, v in parent.items()}}
    rank = {k_obs: 0 for k_obs in parent}
    mst = []

    def find(v):
        """path compression"""
        if parent[v] != v:
            parent[v] = find(parent[v])
        return parent[v]

    def union(a, b):
        """union by rank"""
        root_a, root_b = find(a), find(b)
        if rank[root_a] < rank[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        rank[root_a] += rank[root_a] == rank[root_b]

    def kruskal(dct_edge: dict, num_mst: int) -> None:
        if dct_edge:
            num_edge = len(mst)
            # * notice we sort edges by ABS(bidep) in DESCENDING order
            for (v_l, v_r, s_cond), _ in sorted(
                dct_edge.items(), key=lambda x: abs(x[1]), reverse=True
            ):
                if find((v_l, s_cond)) != find((v_r, s_cond)):
                    mst.append((v_l, v_r, s_cond))
                    union((v_l, s_cond), (v_r, s_cond))
                    num_edge += 1
                if num_edge == num_mst:
                    break

    # ! filter for edges
    s_sub = set()
    dct_edge = dict()
    for i_s in (s_first, set(range(num_dim))):
        s_sub |= i_s
        dct_edge = {
            (v_l, v_r, s_cond): bidep
            for (v_l, v_r, s_cond), bidep in dct_edge_lv.items()
            if ((v_l in s_sub) and (v_r in s_sub))
        }
        dct_edge_lv = {k: v for k, v in dct_edge_lv.items() if k not in dct_edge}
        num_mst = max(0, len(s_sub) - lv - 1)
        kruskal(dct_edge, num_mst=num_mst)
    return mst


def _mst_from_edge_cvine(
    tpl_key_obs: tuple, dct_edge_lv: dict, s_first: set, deq_sim: Deque
) -> tuple:
    """Construct Kruskal's MAXIMUM spanning tree (MST) from bivariate copula edges, restricted to cvine

    :param tpl_key_obs: tuple of keys of (pseudo) observations, each key is a tuple of (vertex, cond set)
    :type tpl_key_obs: tuple
    :param dct_edge_lv: dictionary of edges (vertex_left, vertex_right, cond set)
        and corresponding bivariate dependence metric value
    :type dct_edge_lv: dict
    :param s_first: set of vertices that are kept shallower in the simulation workflow (dynamically updated at each level)
    :type s_first: set
    :param deq_sim: deque of vertices, as simulation workflow (read from right to left,
        as simulated pseudo-obs vertices from shallowest to deepest level) (dynamically updated at each level)
    :type deq_sim: Deque
    :return: list of edges (vertex_left, vertex_right, cond set) in the MST; updated deq_sim; updated s_first
    :rtype: tuple
    """
    # * edge2tree, cvine (MST, restricted to cvine)
    # init dict, the sum of abs bidep for each vertex
    if s_first:
        # ! filter for edges that touch first set vertices
        dct_bidep_abs_sum = {v_s_cond: 0 for v_s_cond in tpl_key_obs if v_s_cond[0] in s_first}
    else:
        dct_bidep_abs_sum = {v_s_cond: 0 for v_s_cond in tpl_key_obs}
    for (v_l, v_r, s_cond), bidep in dct_edge_lv.items():
        # cum sum of abs bidep for each vertex
        if (v_l, s_cond) in dct_bidep_abs_sum:
            dct_bidep_abs_sum[(v_l, s_cond)] += abs(bidep)
        if (v_r, s_cond) in dct_bidep_abs_sum:
            dct_bidep_abs_sum[(v_r, s_cond)] += abs(bidep)
    # center vertex (and its cond set) for cvine at this level
    for (v_c, s_cond_c), _ in sorted(dct_bidep_abs_sum.items(), key=itemgetter(1), reverse=True):
        # ! exclude those already in deq_sim
        if v_c not in deq_sim:
            break
    # record edges that touch the center vertex
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
    # update the first set
    s_first -= {v_c}
    # * mst(cvine), deq_sim, s_first
    return mst, deq_sim, s_first


def _mst_from_edge_dvine(tpl_key_obs: tuple, dct_edge_lv: dict, s_first: set) -> list:
    """Construct Kruskal's MAXIMUM spanning tree (MST) from bivariate copula edges, restricted to dvine.
    For dvine the whole struct (and sim flow) is known after lv0, and this func is only called at lv0.

    :param tpl_key_obs: tuple of keys of (pseudo) observations, each key is a tuple of (vertex, cond set)
    :type tpl_key_obs: tuple
    :param dct_edge_lv: dictionary of edges (vertex_l, vertex_r, cond set) and corresponding bivariate dependence metric value
    :type dct_edge_lv: dict
    :param s_first: set of vertices that are kept shallower in the simulation workflow
    :type s_first: set
    :return: list of edges (vertex_l, vertex_r, cond set) in the MST (lv0)
    :rtype: list
    """
    # * edge2tree, dvine (MST, restricted to dvine)
    # * at lv0, s_cond is known to be empty
    if len(tpl_key_obs) < 3:
        # ! only two vertices
        v_head, v_tail = tpl_key_obs
        v_head, v_tail = v_head[0], v_tail[0]
        mst = [(v_head, v_tail, frozenset())]
    else:
        # ! more than two vertices
        import math

        from networkx import Graph
        from networkx.algorithms.approximation import threshold_accepting_tsp

        dct_cost = {
            (v_l, v_r): math.log1p(1 / max(abs(bidep), 1e-10))
            for (v_l, v_r, s_cond), bidep in dct_edge_lv.items()
        }
        G = Graph()
        G.add_weighted_edges_from([(*k, v) for k, v in dct_cost.items()])
        s_rest = set(G.nodes) - s_first
        if (not s_first) or (not s_rest):
            # ! one set is empty: global TSP
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
        elif len(s_first) in (1, len(tpl_key_obs) - 1):
            # ! one set is a singleton (head-neck-...-tail: add head-neck, drop neck-tail)
            tsp, v_head = (
                (s_first, list(s_rest)[0]) if len(s_rest) == 1 else (s_rest, list(s_first)[0])
            )
            # * TSP on the subgraph without the singleton
            tsp = threshold_accepting_tsp(G.subgraph(tsp), init_cycle="greedy")
            # * loop over the TSP and pick the combination with min cost
            tsp.append(tsp[1])
            cost = math.inf
            for idx, v in enumerate(tsp[1:-1], start=1):
                v_prev, v_next = tsp[idx - 1], tsp[idx + 1]
                # tail given neck
                if (cost_next := dct_cost[tuple(sorted((v, v_next)))]) > (
                    cost_prev := dct_cost[tuple(sorted((v, v_prev)))]
                ):
                    v_nebr, cost_nebr = v_next, cost_next
                else:
                    v_nebr, cost_nebr = v_prev, cost_prev
                # neck
                if (i_cost := dct_cost[tuple(sorted((v_head, v)))] - cost_nebr) < cost:
                    v_tail, v_neck, cost = v_nebr, v, i_cost
            mst = [(*sorted((v_head, v_neck)), frozenset())]
            for idx, v in enumerate(tsp[1:-1], start=1):
                v_l, v_r = sorted((v, tsp[idx + 1]))
                if {v_l, v_r} != {v_neck, v_tail}:
                    mst.append((v_l, v_r, frozenset()))
        else:
            # ! both sets have more than one vertex
            # ! (head-...-neck-body-...-tail, drop head-neck, add neck-body, drop body-tail)
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
                mst.append((*sorted((v_head, v_neck)), frozenset()))
            else:
                for idx, v in enumerate(tsp_first[1:-1], start=1):
                    v_l, v_r = sorted((v, tsp_first[idx + 1]))
                    if {v_l, v_r} != {v_head, v_neck}:
                        mst.append((v_l, v_r, frozenset()))
            if len(tsp_rest) == 4:
                mst.append((*sorted((v_body, v_tail)), frozenset()))
            else:
                for idx, v in enumerate(tsp_rest[1:-1], start=1):
                    v_l, v_r = sorted((v, tsp_rest[idx + 1]))
                    if {v_l, v_r} != {v_body, v_tail}:
                        mst.append((v_l, v_r, frozenset()))
    return mst


def vcp_from_obs(
    obs_mvcp: torch.Tensor,
    is_Dissmann: bool = True,
    matrix: np.ndarray | None = None,
    tpl_first: tuple[int] = tuple(),
    mtd_vine: str = "rvine",
    mtd_bidep: str = "chatterjee_xi",
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
) -> DataVineCop:
    """Construct a vine copula model from multivariate observations,
    with structure prescribed by either Dissmann's (MST per level) method or a given matrix.
    May prioritize some vertices to be first (shallower) in the quantile-regression/conditional-simulation workflow.

    :param obs_mvcp: multivariate observations, of shape (num_obs, num_dim)
    :type obs_mvcp: torch.Tensor
    :param is_Dissmann: whether to use Dissmann's method or follow a given matrix,
        defaults to True; Dissmann, J., Brechmann, E. C., Czado, C., & Kurowicka, D. (2013).
        Selecting and estimating regular vine copulae and application to financial returns.
        Computational Statistics & Data Analysis, 59, 52-69.
    :type is_Dissmann: bool, optional
    :param matrix: a matrix of vine copula structure, of shape (num_dim, num_dim),
        used when is_Dissmann is False, defaults to None
    :type matrix: np.ndarray | None, optional
    :param tpl_first: tuple of vertices to be prioritized (kept shallower) in the cond sim workflow.
        if empty then no priority is set, defaults to tuple()
    :type tpl_first: tuple[int], optional
    :param mtd_vine: one of 'cvine', 'dvine', 'rvine', defaults to "rvine"
    :type mtd_vine: str, optional
    :param mtd_bidep: method to calculate bivariate dependence, one of "kendall_tau", "mutual_info",
        "ferreira_tail_dep_coeff", "chatterjee_xi", "wasserstein_dist_ind", defaults to "chatterjee_xi"
    :type mtd_bidep: str, optional
    :param thresh_trunc: threshold of Kendall's tau independence test, below which we reject independent bicop, defaults to 0.1
    :type thresh_trunc: float, optional
    :param mtd_fit: method to fit bivariate copula, either 'itau' (inverse of tau) or
        'mle' (maximum likelihood estimation); defaults to "itau"
    :type mtd_fit: str, optional
    :param mtd_mle: optimization method for mle as used by scipy.optimize.minimize, defaults to "COBYLA"
    :type mtd_mle: str, optional
    :param mtd_sel: bivariate copula model selection criterion, either 'aic' or 'bic'; defaults to "aic"
    :type mtd_sel: str, optional
    :param tpl_fam: tuple of str as candidate family names to fit bicop,
        could be a subset of ('Clayton', 'Frank', 'Gaussian', 'Gumbel', 'Independent', 'Joe', 'StudentT'),
        defaults to ( "Clayton", "Frank", "Gaussian", "Gumbel", "Independent", "Joe")
    :type tpl_fam: tuple[str, ...], optional
    :param topk: number of best "itau" fit taken into further "mle", used when mtd_fit is "mle"; defaults to 2
    :type topk: int, optional
    :raises ValueError: when mtd_vine is not one of 'cvine', 'dvine', 'rvine'
    :return: a constructed DataVineCop object
    :rtype: DataVineCop
    """
    is_kendall_tau = mtd_bidep == "kendall_tau"
    f_bidep = ENUM_FUNC_BIDEP[mtd_bidep].value
    num_dim = obs_mvcp.shape[1]
    s_first = set(tpl_first)
    s_rest = set(range(num_dim)) - s_first
    # ! an object to record the order of sim (read from right to left,
    # ! as simulated pseudo-obs vertices from shallowest to deepest level)
    deq_sim = deque()
    r_D1 = range(num_dim - 1)
    dct_obs = {_: {} for _ in r_D1}
    # * tree is either from Dissmann (MST on edges, needs to record edges) or from matrix
    if is_Dissmann:
        dct_edge = {_: {} for _ in r_D1}
    dct_tree = {_: {} for _ in r_D1}
    dct_bcp = {_: {} for _ in r_D1}

    def _visit_hfunc(v_down: int, s_down: frozenset) -> None:
        """calc hfunc for pseudo obs and update dct_obs, only when necessary (lazy hfunc)"""
        # * v_down and s_down are from the child pseudo obs
        lv_down = len(s_down)
        # * lv_up and s_up are form the prev lv bcp
        lv_up = lv_down - 1
        for (v_l, v_r, s_up), bcp in dct_bcp[lv_up].items():
            if (v_down in {v_l, v_r}) and s_down.issubset({v_l, v_r} | s_up):
                # ! notice hfunc1 or hfunc2
                if bcp.fam == "Independent":
                    dct_obs[lv_down][(v_down, s_down)] = dct_obs[lv_up][(v_down, s_up)]
                else:
                    dct_obs[lv_down][(v_down, s_down)] = (
                        bcp.hfunc2 if v_down == v_l else bcp.hfunc1
                    )(
                        obs=torch.hstack(
                            [
                                dct_obs[lv_up][(v_l, s_up)],
                                dct_obs[lv_up][(v_r, s_up)],
                            ]
                        )
                    )

    for lv in r_D1:
        # * lv_0 obs, preprocess to append an empty frozenset (s_cond)
        if lv == 0:
            dct_obs[0] = {(idx, frozenset()): obs_mvcp[:, [idx]] for idx in range(num_dim)}
        if is_Dissmann:
            # * obs2edge, list possible edges that connect two pseudo obs, calc f_bidep
            tpl_key_obs = dct_obs[lv].keys()
            dct_s_cond_v = defaultdict(list)
            for v, s_cond in tpl_key_obs:
                dct_s_cond_v[s_cond].append(v)
            # ! proximity condition: only those obs with same 'cond set' (the frozen set) can have edges
            for s_cond, lst_v in dct_s_cond_v.items():
                if len(lst_v) > 1:
                    for v_l, v_r in combinations(lst_v, 2):
                        # ! sorted !
                        v_l, v_r = sorted((v_l, v_r))
                        if dct_obs[lv][v_l, s_cond] is None:
                            _visit_hfunc(v_l, s_cond)
                        if dct_obs[lv][v_r, s_cond] is None:
                            _visit_hfunc(v_r, s_cond)
                        if is_kendall_tau:
                            dct_edge[lv][(v_l, v_r, s_cond)] = f_bidep(
                                x=dct_obs[lv][v_l, s_cond],
                                y=dct_obs[lv][v_r, s_cond],
                            )[0]
                        else:
                            dct_edge[lv][(v_l, v_r, s_cond)] = f_bidep(
                                x=dct_obs[lv][v_l, s_cond],
                                y=dct_obs[lv][v_r, s_cond],
                            )
            if mtd_vine == "dvine":
                # * edge2tree, dvine
                # ! for dvine, the whole struct is known after lv0
                if lv == 0:
                    mst = _mst_from_edge_dvine(
                        tpl_key_obs=tpl_key_obs,
                        dct_edge_lv=dct_edge[lv],
                        s_first=s_first,
                    )
                    dct_tree[lv] = {key_edge: dct_edge[lv][key_edge] for key_edge in mst}
                else:
                    dct_tree[lv] = dct_edge[lv]
            elif mtd_vine == "cvine":
                # * edge2tree, cvine
                # ! for cvine, at each lv the center vertex is the one with the largest sum of abs bidep
                mst, deq_sim, s_first = _mst_from_edge_cvine(
                    tpl_key_obs=tpl_key_obs,
                    dct_edge_lv=dct_edge[lv],
                    s_first=s_first,
                    deq_sim=deq_sim,
                )
                dct_tree[lv] = {key_edge: dct_edge[lv][key_edge] for key_edge in mst}
            elif mtd_vine == "rvine":
                # * edge2tree, rvine
                mst = _mst_from_edge_rvine(
                    tpl_key_obs=tpl_key_obs,
                    dct_edge_lv=dct_edge[lv],
                    s_first=s_first,
                    lv=lv,
                    num_dim=num_dim,
                )
                dct_tree[lv] = {key_edge: dct_edge[lv][key_edge] for key_edge in mst}
            else:
                raise ValueError("mtd_vine must be one of 'cvine', 'dvine', 'rvine'")
        else:
            # * tree structure is inferred from matrix, if not Dissmann
            for idx in range(num_dim - lv - 1):
                # ! sorted !
                v_l, v_r = sorted((matrix[idx, idx], matrix[idx, num_dim - lv - 1]))
                s_cond = frozenset(matrix[idx, (num_dim - lv) :])
                if dct_obs[lv][v_l, s_cond] is None:
                    _visit_hfunc(v_down=v_l, s_down=s_cond)
                if dct_obs[lv][v_r, s_cond] is None:
                    _visit_hfunc(v_down=v_r, s_down=s_cond)
                if is_kendall_tau:
                    dct_tree[lv][(v_l, v_r, s_cond)] = f_bidep(
                        x=dct_obs[lv][(v_l, s_cond)],
                        y=dct_obs[lv][(v_r, s_cond)],
                    )[0]
                else:
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
                thresh_trunc=thresh_trunc,
                mtd_fit=mtd_fit,
                mtd_mle=mtd_mle,
                mtd_sel=mtd_sel,
                tpl_fam=tpl_fam,
                topk=topk,
            )
            i_lv_next = lv + 1
            if i_lv_next < num_dim - 1:
                dct_obs[i_lv_next][v_r, s_cond | {v_l}] = None
                dct_obs[i_lv_next][v_l, s_cond | {v_r}] = None
        # ! garbage collection
        if is_Dissmann:
            del dct_edge[lv]
        if lv > 0:
            del dct_obs[lv - 1]
    # * for cvine, deq_sim is known and s_first is empty now
    # * for dvine/rvine, deq_sim is empty and to be arranged
    if not deq_sim:
        # ! sequentially "peel off" paths from the vine
        # the deepest lv, symmetric
        deq_sim = [*dct_tree[num_dim - 2]][0][:2]
        deq_sim = deque(v for v in deq_sim if v in s_rest)
        while (lv := num_dim - 2 - len(deq_sim)) > -1:
            for v_l, v_r, _ in [*dct_tree[lv]]:
                # * locate the virgin bicop vertex at this lv
                if (v_l not in deq_sim) and (v_r not in deq_sim):
                    # * those in s_rest are prioritized
                    if (is_v_l_deeper := v_l in s_rest) ^ (v_r in s_rest):
                        deq_sim.append(v_l if is_v_l_deeper else v_r)
                    else:
                        # ! select the one with less hfunc calls
                        deq_sim.append(
                            v_l
                            if ref_count_hfunc(dct_tree=dct_tree, tpl_sim=tuple(deq_sim) + (v_l,))[
                                -1
                            ]
                            <= ref_count_hfunc(dct_tree=dct_tree, tpl_sim=tuple(deq_sim) + (v_r,))[
                                -1
                            ]
                            else v_r
                        )
                    break
            if not lv:
                # the top lv
                deq_sim.extend({v_l, v_r} - {*deq_sim})
                break

    return DataVineCop(
        dct_bcp=dct_bcp,
        dct_tree=dct_tree,
        tpl_sim=tuple(deq_sim),
        mtd_bidep=mtd_bidep,
    )


def vcp_from_json(f_path: Path = Path("./vcp.json")) -> DataVineCop:
    """load a DataVineCop from a json file

    :param f_path: path to the json file, defaults to Path("./vcp.json")
    :type f_path: Path, optional
    :return: a DataVineCop object
    :rtype: DataVineCop
    """
    with open(f_path, "r") as file:
        tmp_json = json.load(file)
    #
    dct_bcp = defaultdict(dict)
    for lv, i_dct in tmp_json["dct_bcp"].items():
        for key, val in i_dct.items():
            v_l, v_r, s_cond = literal_eval(key)
            s_cond = frozenset(s_cond)
            val["par"] = tuple(val["par"])
            dct_bcp[int(lv)][(v_l, v_r, s_cond)] = DataBiCop(**val)
    tmp_json["dct_bcp"] = dict(dct_bcp)
    #
    dct_tree = defaultdict(dict)
    for lv, i_dct in tmp_json["dct_tree"].items():
        for key, val in i_dct.items():
            v_l, v_r, s_cond = literal_eval(key)
            s_cond = frozenset(s_cond)
            dct_tree[int(lv)][(v_l, v_r, s_cond)] = val
    tmp_json["dct_tree"] = dict(dct_tree)
    #
    tmp_json["tpl_sim"] = tuple(tmp_json["tpl_sim"])
    #
    return DataVineCop(**tmp_json)


def vcp_from_pth(f_path: Path = Path("./vcp.pth")) -> DataVineCop:
    """load a DataVineCop from a pth file

    :param f_path: path to the pth file, defaults to Path("./vcp.pth")
    :type f_path: Path, optional
    :return: a DataVineCop object
    :rtype: DataVineCop
    """
    with open(f_path, "rb") as file:
        obj = torch.load(file)
    return obj
