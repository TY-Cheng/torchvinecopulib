from __future__ import annotations

import copy
import heapq
from collections import Counter, defaultdict

__all__ = ["_ref_count_hfunc_impl", "VineCopStructureMixin"]


def _ref_count_hfunc_impl(
    num_dim: int,
    struct_obs: list[dict[tuple[int, ...], str]],
    sample_order: tuple[int, ...],
) -> tuple[dict[tuple[int, ...], int], list[tuple[int, ...]], int]:
    missing = set(range(num_dim)) - set(sample_order)
    lst_source = []
    for idx, v in enumerate(sample_order):
        s = set(sample_order[idx + 1 :]) | missing
        lst_source.append((v, *sorted(s)))
    for v in missing:
        lst_source.append((v,))
    lst_source.reverse()
    ref_cnt = Counter()
    num_hfunc = 0

    def _visit(v_s: tuple[int, ...], is_hinv: bool = False):
        nonlocal num_hfunc
        if len(v_s) == 1:
            ref_cnt[v_s] += 1
            return
        v_down, *s_down = v_s
        v_l, v_r = map(int, struct_obs[len(s_down)][v_s].split(","))
        s_up = tuple(sorted(set(s_down) - {v_l, v_r}))
        frontier = (
            [(v_l if v_down == v_r else v_r, *s_up)] if is_hinv else [(v_l, *s_up), (v_r, *s_up)]
        )
        for v_s_parent in frontier:
            if ref_cnt[v_s_parent] == 0:
                _visit(v_s=v_s_parent, is_hinv=False)
                num_hfunc += 1
        frontier = [(v_l, *s_up), (v_r, *s_up), (v_down, *s_down)]
        for v_s_parent in frontier:
            ref_cnt[v_s_parent] += 1
        if is_hinv:
            return (v_down, *s_up)

    for v_s in lst_source:
        if len(v_s) == 1:
            ref_cnt[v_s] += 1
        while len(v_s) > 1:
            v_s = _visit(v_s=v_s, is_hinv=True)
    return dict(ref_cnt), lst_source, num_hfunc


class VineCopStructureMixin:
    @staticmethod
    def ref_count_hfunc(
        num_dim: int, struct_obs: list, sample_order: tuple
    ) -> tuple[dict, list, int]:
        """Count references of pseudo-obs, identify source vertices, and count number of hfuncs.

        Args:
            num_dim (int): number of dimensions in the vine.
            struct_obs (list): structure of pseudo observations. (parents)
            sample_order (tuple): sampling order.

        Returns:
            tuple[dict, list, int]: reference counts of pseudo-obs, list of source vertices, and number of hfuncs.
        """
        return _ref_count_hfunc_impl(
            num_dim=num_dim, struct_obs=struct_obs, sample_order=sample_order
        )

    def _sample_order(self) -> tuple[int, ...]:
        """Schedule an optimized sampling order to minimize h‐function calls.

        Returns:
            tuple[int, ...]: New ``sample_order`` of variable indices for inverse Rosenblatt sampling.
        """
        last_tree_vertex = set(range(self.num_dim)) - set(self.first_tree_vertex)
        sample_order = []
        for v_s_parent in self.struct_obs[::-1]:
            cost_best = float("inf")
            cand_v = set()
            for v_s, cond_ed in v_s_parent.items():
                if cond_ed:
                    # * not yet top lv
                    v_l, v_r = map(int, cond_ed.split(","))
                    if v_l not in sample_order and v_r not in sample_order:
                        cand_v.add(v_l)
                        cand_v.add(v_r)
                elif v_s[0] not in sample_order:
                    # * top lv, only one choice
                    cand_v.add(v_s[0])
            # ! prioritize those not in first_tree_vertex
            cand_v_last = cand_v & last_tree_vertex
            if cand_v_last:
                cand_v = cand_v_last
            for v in sorted(cand_v):
                _, _, cost = self.ref_count_hfunc(
                    num_dim=self.num_dim,
                    struct_obs=self.struct_obs,
                    sample_order=sample_order + [v],
                )
                if cost < cost_best:
                    cost_best = cost
                    v_best = v
            sample_order.append(v_best)
        self.sample_order = tuple(sample_order)

    def _mst_from_edge_dvine(self, edge_weight_lv: dict) -> None:
        # * edge2tree, dvine (MST, restricted to dvine)
        # * TSP with precedence constraints (clustered TSP), only called at lv-0
        # ! all s have to be empty for level‑0 D‑vine
        edge_list = [((v_l, v_r), -abs(w)) for (v_l, v_r, *s), w in edge_weight_lv.items()]
        edge_list.sort(key=lambda x: x[1])
        parent = list(range(self.num_dim))
        degree = [0] * self.num_dim

        def find(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx
                return True
            return False

        def grow(edge_list, target):
            for (v_l, v_r), _ in edge_list:
                if len(path) >= target:
                    break
                if degree[v_l] < 2 and degree[v_r] < 2 and union(v_l, v_r):
                    path.append((v_l, v_r))
                    degree[v_l] += 1
                    degree[v_r] += 1

        path: list[tuple[int, int]] = []
        cand_v = list(self.first_tree_vertex)
        if len(cand_v) > 1:
            cand_e = [e for e in edge_list if e[0][0] in cand_v and e[0][1] in cand_v]
            grow(edge_list=cand_e, target=len(cand_v) - 1)
        grow(edge_list=edge_list, target=self.num_dim - 1)
        # * canonicalize ordering
        return [(min(v_l, v_r), max(v_l, v_r)) for v_l, v_r in path]

    def _mst_from_edge_cvine(self, edge_weight_lv: dict) -> None:
        # * edge2tree, cvine (MST, restricted to cvine)
        # * accumulate |weight| for every pseudo obs vertex (v,s)
        score = defaultdict(float)
        for (v_l, v_r, *s), w in edge_weight_lv.items():
            s = tuple(s)
            w = abs(w)
            score[v_l, s] += w
            score[v_r, s] += w
        # * restrict candidate set if precedence given
        # * fallback if impossible
        cand_v = None
        if self.first_tree_vertex:
            cand_v = {v_s for v_s in score if v_s[0] in self.first_tree_vertex}
        if not cand_v:
            cand_v = set(score)
        v_c, s_c = max(cand_v, key=lambda x: score[x])
        mst = [
            (v_l, v_r, tuple(s))
            for (v_l, v_r, *s), _ in edge_weight_lv.items()
            if tuple(s) == s_c and (v_l == v_c or v_r == v_c)
        ]
        # * canonicalize ordering
        return [(min(v_l, v_r), max(v_l, v_r), *s) for v_l, v_r, s in mst]

    def _mst_from_edge_rvine(self, edge_weight_lv: dict) -> None:
        # * edge2tree, rvine (Kruskal's MST, disjoint set/ union find)
        # * lr_s[2:] is cond_ing set
        edge_weight_lv = copy.deepcopy(edge_weight_lv)
        lv = len(next(iter(edge_weight_lv))[2:])
        # ! pseudo obs vertices point to parent bicop vertices (constraint set)
        parent = {}
        for lr_s, cond_ed in self.struct_obs[lv].items():
            parent[lr_s] = (
                frozenset(lr_s + tuple(map(int, cond_ed.split(","))))
                if lv > 0
                else frozenset(lr_s)
            )
        # * bicop vertices point to themselves
        parent.update({r: r for r in parent.values()})
        rank = {r: 0 for r in parent}
        mst = []

        def find(v):
            """
            Path compression.
            """
            if parent[v] != v:
                parent[v] = find(parent[v])
            return parent[v]

        def union(x, y):
            """
            Union by rank.
            """
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            rank[rx] += rank[rx] == rank[ry]
            return True

        def kruskal(cand_e_w: dict, num_mst: int) -> None:
            if cand_e_w:
                # ! min heap, by -ABS(bidep) in ASCENDING order
                heap_bidep_abs = [(-abs(bidep), lr_s) for lr_s, bidep in cand_e_w.items()]
                heapq.heapify(heap_bidep_abs)
                while len(mst) < num_mst:
                    _, lr_s = heapq.heappop(heap_bidep_abs)
                    v_l, v_r, *s = lr_s
                    if union(find((v_l, *s)), find((v_r, *s))):
                        mst.append(lr_s)

        # * gradually grow the vine, filter for edges
        cand_v = set()
        for step_v in (set(self.first_tree_vertex), set(range(self.num_dim))):
            cand_v |= step_v
            cand_e_w = {
                # * pop update edge_weight_lv
                e: edge_weight_lv.pop(e)
                for e in list(edge_weight_lv)
                # * edges with both vertices in cand_v
                if e[0] in cand_v and e[1] in cand_v
            }
            kruskal(
                cand_e_w=cand_e_w,
                num_mst=max(
                    0, len(cand_v) - lv - 1
                ),  # * number of edges in the MST, at this stage
            )
        return mst
