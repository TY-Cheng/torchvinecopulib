import heapq
import json
import math
from ast import literal_eval
from collections import defaultdict, deque
from itertools import combinations, product
from pathlib import Path
from random import seed as r_seed
from typing import Deque

import numpy as np
import torch
from networkx import Graph
from networkx.algorithms.approximation import threshold_accepting_tsp

from ..bicop import ENUM_FAM_BICOP, DataBiCop, SET_FAMnROT, bcp_from_obs
from ..util import ENUM_FUNC_BIDEP, ref_count_hfunc
from ._data_vcp import DataVineCop


def _mst_from_edge_rvine(
    tpl_key_obs: tuple, dct_edge_lv: dict, s_first: set, lv: int, num_dim: int
) -> list:
    """Construct Kruskal's MAXIMUM spanning tree (MST) from bivariate copula edges,
    restricted to rvine, using modified disjoint set/ union find.

    :param tpl_key_obs: tuple of keys of (pseudo) observations,
        each key is a tuple of (vertex, cond set)
    :type tpl_key_obs: tuple
    :param dct_edge_lv: dictionary of edges (vertex_left, vertex_right, cond_set)
        and corresponding bivariate dependence metric value
    :type dct_edge_lv: dict
    :param s_first: set of vertices that are kept shallower
        in the simulation workflow (dynamically updated at each level)
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
    parent.update({v: v for _, v in parent.items()})
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
        if root_a == root_b:
            return False
        if rank[root_a] < rank[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        rank[root_a] += rank[root_a] == rank[root_b]
        return True

    def kruskal(dct_edge: dict, num_mst: int) -> None:
        # * skip those forming cycles
        if dct_edge:
            # ! min heap, by -ABS(bidep) in ASCENDING order
            heap_bidep_abs = [
                (-abs(bidep), v_l, v_r, s_cond) for (v_l, v_r, s_cond), bidep in dct_edge.items()
            ]
            heapq.heapify(heap_bidep_abs)
            while len(mst) < num_mst:
                _, v_l, v_r, s_cond = heapq.heappop(heap_bidep_abs)
                if union(find((v_l, s_cond)), find((v_r, s_cond))):
                    mst.append((v_l, v_r, s_cond))

    # ! filter for edges
    s_sub = set()
    dct_edge = {}
    for i_s in (s_first, set(range(num_dim))):
        # * gradually grow the vine
        s_sub |= i_s
        # * edges with both vertices in s_sub, from dct_edge_lv
        dct_edge = {
            (v_l, v_r, s_cond): bidep
            for (v_l, v_r, s_cond), bidep in dct_edge_lv.items()
            if ((v_l in s_sub) and (v_r in s_sub))
        }
        # * update dct_edge_lv, those with at least one vertex not in s_sub
        dct_edge_lv = {k: v for k, v in dct_edge_lv.items() if k not in dct_edge}
        # * number of edges in the MST, at this stage
        num_mst = max(0, len(s_sub) - lv - 1)
        kruskal(dct_edge, num_mst=num_mst)
    return mst


def _mst_from_edge_cvine(
    tpl_key_obs: tuple, dct_edge_lv: dict, s_first: set, deq_sim: Deque
) -> tuple:
    """Construct Kruskal's MAXIMUM spanning tree (MST) from bivariate copula edges,
        restricted to cvine

    :param tpl_key_obs: tuple of keys of (pseudo) observations,
        each key is a tuple of (vertex, cond set)
    :type tpl_key_obs: tuple
    :param dct_edge_lv: dictionary of edges (vertex_left, vertex_right, cond set)
        and corresponding bivariate dependence metric value
    :type dct_edge_lv: dict
    :param s_first: set of vertices that are kept shallower
        in the simulation workflow (dynamically updated at each level)
    :type s_first: set
    :param deq_sim: deque of vertices, as simulation workflow
        (read from right to left, as simulated pseudo-obs vertices from
        shallowest to deepest level) (dynamically updated at each level)
    :type deq_sim: Deque
    :return: list of edges (vertex_left, vertex_right, cond set) in the MST;
        updated deq_sim; updated s_first
    :rtype: tuple
    """
    # * edge2tree, cvine (MST, restricted to cvine)
    # * init dict, the sum of abs bidep for each vertex
    if s_first:
        # ! filter for edges that touch first set vertices
        dct_bidep_abs_sum = {v_s_cond: 0 for v_s_cond in tpl_key_obs if v_s_cond[0] in s_first}
    else:
        dct_bidep_abs_sum = {v_s_cond: 0 for v_s_cond in tpl_key_obs}
    for (v_l, v_r, s_cond), bidep in dct_edge_lv.items():
        # * cum sum of abs bidep for each vertex
        if (v_l, s_cond) in dct_bidep_abs_sum:
            dct_bidep_abs_sum[(v_l, s_cond)] += abs(bidep)
        if (v_r, s_cond) in dct_bidep_abs_sum:
            dct_bidep_abs_sum[(v_r, s_cond)] += abs(bidep)
    # * center vertex (and its cond set) for cvine at this level
    # ! min heap, by -ABS(bidep) in ASCENDING order
    heap_bidep_abs_sum = [
        (-bidep_abs_sum, v_s_cond) for v_s_cond, bidep_abs_sum in dct_bidep_abs_sum.items()
    ]
    heapq.heapify(heap_bidep_abs_sum)
    v_c = None
    while v_c is None:
        _, (v_c, s_cond_c) = heapq.heappop(heap_bidep_abs_sum)
        # ! exclude those already in deq_sim
        if v_c in deq_sim:
            v_c = None
    # * record edges that touch the center vertex
    mst = [
        (v_l, v_r, s_cond)
        for (v_l, v_r, s_cond) in dct_edge_lv
        if ((s_cond == s_cond_c) and ((v_c == v_l) or (v_c == v_r)))
    ]
    # * update the deq_sim, let those sim first be last in the deq_sim
    deq_sim.appendleft(v_c)
    if len(mst) == 1:
        # * for the last lv, only one edge left, appendleft the other vertex into deq_sim
        deq_sim.appendleft(mst[0][0] if v_c != mst[0][0] else mst[0][1])
    # * update the first set
    s_first -= {v_c}
    # * mst(cvine), deq_sim, s_first
    return mst, deq_sim, s_first


def _mst_from_edge_dvine(tpl_key_obs: tuple, dct_edge_lv: dict, s_first: set) -> list:
    """Construct Kruskal's MAXIMUM spanning tree (MST) from bivariate copula edges,
        restricted to dvine.
    For dvine the whole struct (and sim flow) is known after lv0,
        and this func is only called at lv0.
    TSP with precedence constraints (clustered TSP).

    :param tpl_key_obs: tuple of keys of (pseudo) observations,
        each key is a tuple of (vertex, cond set)
    :type tpl_key_obs: tuple
    :param dct_edge_lv: dictionary of edges (vertex_l, vertex_r, cond set)
        and corresponding bivariate dependence metric value
    :type dct_edge_lv: dict
    :param s_first: set of vertices that are kept
        shallower in the simulation workflow
    :type s_first: set
    :return: list of edges (vertex_l, vertex_r, cond set) in the MST (lv0)
    :rtype: list
    """

    def _edges_from_tsp(tsp: list) -> set:
        return {tuple(sorted((tsp[idx], tsp[idx - 1]))) for idx in range(1, len(tsp))}

    # * edge2tree, dvine (MST, restricted to dvine)
    # * at lv0, s_cond is known to be empty
    if len(tpl_key_obs) < 3:
        # ! only two vertices
        return [(tpl_key_obs[0], tpl_key_obs[1], frozenset())]
    dct_cost = {
        (v_l, v_r): math.log1p(1 / max(abs(bidep), 1e-10))
        for (v_l, v_r, s_cond), bidep in dct_edge_lv.items()
    }
    # * weighted undirected graph
    graph_nx = Graph()
    graph_nx.add_weighted_edges_from([(*k, v) for k, v in dct_cost.items()])
    s_rest = set(graph_nx.nodes) - s_first
    if len(s_first) <= 1 or len(s_rest) <= 1:
        # ! one set is empty or singleton: global TSP
        tsp = threshold_accepting_tsp(graph_nx, init_cycle="greedy")
        mst = []
        tmp_cost, tmp_drop = -math.inf, None
        for idx in range(1, len(tsp)):
            v_lr = tuple(sorted((tsp[idx], tsp[idx - 1])))
            v_lr_cost = dct_cost[v_lr]
            mst.append(v_lr)
            # ! drop the edge with max cost and (at least) one vertex not in s_first
            if (set(v_lr) - s_first) and (v_lr_cost > tmp_cost):
                tmp_cost, tmp_drop = v_lr_cost, v_lr
        mst.remove(tmp_drop)
    else:
        # ! both sets have more than two vertices, TSP with precedence constraints (clustered TSP)
        tsp_first = threshold_accepting_tsp(graph_nx.subgraph(s_first), init_cycle="greedy")
        tsp_rest = threshold_accepting_tsp(graph_nx.subgraph(s_rest), init_cycle="greedy")
        edge_first = _edges_from_tsp(tsp_first)
        edge_rest = _edges_from_tsp(tsp_rest)
        edge_2tsp = edge_first | edge_rest
        cost_2tsp = sum(dct_cost[key] for key in edge_2tsp)
        cost_mst = math.inf
        # ! add, add, (drop, drop), drop
        for drop_first, drop_rest in product(edge_first, edge_rest):
            for edge_add_1, edge_add_2 in [*zip(drop_first, drop_rest)], [
                *zip(drop_rest, drop_first[::-1])
            ]:
                # ! add two edges to chain the two TSPs
                edge_add_1 = tuple(sorted(edge_add_1))
                edge_add_2 = tuple(sorted(edge_add_2))
                tmp_cost = cost_2tsp + dct_cost[edge_add_1] + dct_cost[edge_add_2]
                tmp_mst = edge_2tsp | {edge_add_1, edge_add_2}
                if len(edge_first) > 1:
                    # ! drop one edge from a TSP if it has more than one edge
                    tmp_cost -= dct_cost[drop_first]
                    tmp_mst.remove(drop_first)
                if len(edge_rest) > 1:
                    # ! drop one edge from a TSP if it has more than one edge
                    tmp_cost -= dct_cost[drop_rest]
                    tmp_mst.remove(drop_rest)
                # ! drop the edge with max cost and (at least) one vertex not in s_first
                drop_non_first = max(
                    (k for k in tmp_mst if (k[0] not in s_first) or (k[1] not in s_first)),
                    key=lambda x: dct_cost[x],
                )
                tmp_cost -= dct_cost[drop_non_first]
                tmp_mst.remove(drop_non_first)
                if tmp_cost < cost_mst:
                    cost_mst = tmp_cost
                    mst = tmp_mst
    # * s_cond is known empty at lv0
    mst = [(v_l, v_r, frozenset()) for (v_l, v_r) in mst]
    return mst


def _tpl_sim(deq_sim: deque, dct_tree: dict, s_rest: set) -> tuple:
    """arrange the sampling order tuple, which from right to left
        indicates source vertices from shallowest to deepest level during simulation.

    :param deq_sim: should be full for cvine, empty for dvine/rvine
    :type deq_sim: deque
    :param dct_tree: tree
    :type dct_tree: dict
    :param s_rest: set of indices to rest in the deepest levels
    :type s_rest: set
    :return: a sampling order tuple
    :rtype: tuple
    """
    if not deq_sim:
        lv_max = max(dct_tree)
        # ! sequentially "peel off" paths from the vine
        while (lv := lv_max - len(deq_sim)) > -1:
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
                    if lv == 0:
                        # * the top lv
                        deq_sim.append(v_l if v_l not in deq_sim else v_r)
                    break
    return tuple(deq_sim)


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
    r_dim_1 = range(num_dim - 1)
    dct_obs = {_: {} for _ in r_dim_1}
    # * tree is either from Dissmann (MST on edges, needs to record edges) or from matrix
    if is_Dissmann:
        dct_edge = {_: {} for _ in r_dim_1}
    dct_tree = {_: {} for _ in r_dim_1}
    dct_bcp = {_: {} for _ in r_dim_1}

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

    for lv in r_dim_1:
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
                        for v in (v_l, v_r):
                            # can be tensor or None
                            if dct_obs[lv][v, s_cond] is None:
                                _visit_hfunc(v_down=v, s_down=s_cond)
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
                # can be tensor or None
                if dct_obs[lv][v_l, s_cond] is None:
                    _visit_hfunc(v_down=v_l, s_down=s_cond)
                # can be tensor or None
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
    tpl_sim = _tpl_sim(deq_sim=deq_sim, dct_tree=dct_tree, s_rest=s_rest)
    return DataVineCop(
        dct_bcp=dct_bcp,
        dct_tree=dct_tree,
        tpl_sim=tpl_sim,
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
    tmp_json["tpl_sim"] = tuple(tmp_json["tpl_sim"])
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


def vcp_from_sim(num_dim: int, seed: int = 0) -> DataVineCop:
    """Simulate a DataVineCop object.
        It constructs the vine copula by generating correlation matrices,
        fitting bivariate copulas, and building the vine structure through a series of steps.

    :param num_dim: The number of dimensions for the vine copula.
    :type num_dim: int
    :param seed: The random seed for reproducibility.
    :type seed: int
    :return: An object representing the simulated vine copula.
    :rtype: DataVineCop
    """

    def _corr_from_sim(num_dim: int, seed: int):
        r_seed(seed)
        np.random.seed(seed)
        mat = np.random.uniform(-1, 1, size=(num_dim, num_dim))
        mat = mat @ mat.T
        mat_scale = np.diag(1 / np.sqrt(mat.diagonal()))
        return mat_scale @ mat @ mat_scale

    r_seed(seed)
    np.random.seed(seed)
    lst_famrot = [famrot for famrot in SET_FAMnROT if famrot[0] != "StudentT"]
    r_dim_1 = range(num_dim - 1)
    dct_obs = {_: {} for _ in r_dim_1}
    dct_edge = {_: {} for _ in r_dim_1}
    dct_tree = {_: {} for _ in r_dim_1}
    dct_bcp = {_: {} for _ in r_dim_1}
    deq_sim = deque()
    for lv in r_dim_1:
        # * lv_0 obs, preprocess to append an empty frozenset (s_cond)
        if lv == 0:
            dct_obs[0] = {(idx, frozenset()): None for idx in range(num_dim)}
        mat_corr = _corr_from_sim(num_dim, seed=seed + lv)
        # * obs2edge, list possible edges that connect two pseudo obs, calc f_bidep
        tpl_key_obs = dct_obs[lv].keys()
        dct_s_cond_v = defaultdict(list)
        for v, s_cond in tpl_key_obs:
            dct_s_cond_v[s_cond].append(v)
        # ! proximity condition: only those obs with same 'cond set' (the frozen set) can have edges
        for s_cond, lst_v in dct_s_cond_v.items():
            if len(lst_v) > 1:
                for v_l, v_r in combinations(lst_v, 2):
                    v_l, v_r = sorted((v_l, v_r))
                    for v in (v_l, v_r):
                        dct_edge[lv][(v_l, v_r, s_cond)] = mat_corr[v_l, v_r]
        # * edge2tree, rvine
        mst = _mst_from_edge_rvine(
            tpl_key_obs=tpl_key_obs,
            dct_edge_lv=dct_edge[lv],
            s_first=set(),
            lv=lv,
            num_dim=num_dim,
        )
        dct_tree[lv] = {key_edge: dct_edge[lv][key_edge] for key_edge in mst}
        # * tree2bicop, fit bicop & record key of potential pseudo obs for next lv (lazy hfunc later)
        for (v_l, v_r, s_cond), bidep in dct_tree[lv].items():
            fam, rot = lst_famrot[np.random.choice(len(lst_famrot))]
            if bidep <= 0.05:
                dct_bcp[lv][(v_l, v_r, s_cond)] = DataBiCop(
                    fam="Independent",
                    par=tuple(),
                    rot=0,
                )
            elif fam == "StudentT":
                dct_bcp[lv][(v_l, v_r, s_cond)] = DataBiCop(
                    fam=fam,
                    par=(
                        ENUM_FAM_BICOP["Gaussian"].value.tau2par(bidep, rot=rot)
                        + (np.random.uniform(1, 50),)
                    ),
                    rot=rot,
                )
            else:
                dct_bcp[lv][(v_l, v_r, s_cond)] = DataBiCop(
                    fam=fam,
                    par=ENUM_FAM_BICOP[fam].value.tau2par(bidep, rot=rot),
                    rot=rot,
                )
            lv_next = lv + 1
            if lv_next < num_dim - 1:
                dct_obs[lv_next][(v_l, s_cond | {v_r})] = None
                dct_obs[lv_next][(v_r, s_cond | {v_l})] = None
    tpl_sim = _tpl_sim(deq_sim=deq_sim, dct_tree=dct_tree, s_rest=set())
    return vcp_from_obs(
        obs_mvcp=DataVineCop(
            dct_bcp=dct_bcp,
            dct_tree=dct_tree,
            tpl_sim=tpl_sim,
            mtd_bidep="kendall_tau",
        ).sim(num_sim=5000, seed=seed),
        tpl_first=tuple(deq_sim)[-(num_dim // 2) :],
        mtd_bidep="kendall_tau",
    )
