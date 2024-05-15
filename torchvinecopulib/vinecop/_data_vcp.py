import json
import math
from abc import ABC
from dataclasses import asdict, dataclass
from operator import itemgetter
from pathlib import Path
from pprint import pformat

import numpy as np
import torch


@dataclass(slots=True, frozen=True, kw_only=True)
class DataVineCop(ABC):
    """Dataclass for a vine copula model"""

    dct_bcp: dict
    """bivariate copulas, stored as {level: {(vertex_left, vertex_right, frozenset_cond): DataBiCop}}"""
    dct_tree: dict
    """
    bivariate dependency measures of edges in trees,
    stored as {level: {(vertex_left, vertex_right, frozenset_cond): bidep}}
    """
    tpl_sim: tuple
    """
    the source vertices (pseudo-obs) of simulation paths, read from right to left;
    some vertices can be given as simulated at the beginning of each simulation workflow
    """
    mtd_bidep: str
    """method to calculate bivariate dependence"""

    @property
    def aic(self) -> float:
        """

        :param self: an instance of the DataVineCop dataclass
        :return: Akaike information criterion (AIC)
        :rtype: float
        """
        return 2.0 * (self.num_par + self.negloglik)

    @property
    def bic(self) -> float:
        """

        :param self: an instance of the DataVineCop dataclass
        :return: Bayesian information criterion (BIC)
        :rtype: float
        """
        return 2.0 * self.negloglik + self.num_par * math.log(self.num_obs)

    @property
    def matrix(self) -> np.array:
        """structure matrix: upper triangular, in row-major order
        (each row has a bicop as: vertex_left,...,vertex_right;set_cond)

        :return: structure matrix
        :rtype: np.array
        """
        # NumPy arrays, like arrays in C, are stored in row-major order by default.
        # In a 2D array, the data is stored row by row: elements in the same row are stored in adjacent memory locations.
        mat = []
        lst_diag = []
        # * levels in reverse order
        for idx, lv in enumerate(sorted(self.dct_tree, reverse=True)):
            v_diag = None
            lst_nebr = [-1 for _ in range(idx)]
            for i_lv in range(lv, -1, -1):
                for v_l, v_r, _ in self.dct_tree[i_lv]:
                    if (v_l not in lst_diag) and (v_r not in lst_diag):
                        if v_diag is None:
                            # ! pick the node with smaller index (v_l < v_r), then mat <-> structure is bijection
                            v_diag = v_l
                            lst_nebr.append(v_diag)
                        if v_diag in (v_l, v_r):
                            lst_nebr.append(v_l if v_diag == v_r else v_r)
            lst_diag.append(v_diag)
            mat.append(lst_nebr)
        # ! append the last node
        mat.append([-1 for _ in range(idx + 1)] + [mat[-1][-1]])
        return np.array(mat)

    @property
    def negloglik(self) -> float:
        """nll, as sum of negative log likelihoods of all bivariate copulas

        :return: negative log likelihood
        :rtype: float
        """
        return sum([bcp.negloglik for dct_lv in self.dct_bcp.values() for bcp in dct_lv.values()])

    @property
    def num_dim(self) -> int:
        """number of dimensions"""
        return len(self.dct_tree) + 1

    @property
    def num_obs(self) -> int:
        """number of observations"""
        return [bcp.num_obs for bcp in self.dct_bcp[0].values()][0]

    @property
    def num_par(self) -> int:
        """number of parameters"""
        return sum([bcp.num_par for dct_lv in self.dct_bcp.values() for bcp in dct_lv.values()])

    def _loc_bcp(self, v_down: int, s_down: frozenset) -> tuple:
        """locate the bicop on upper level that generates this pseudo obs

        :param v_down: given vertex on lower level
        :type v_down: int
        :param s_down: given cond set on lower level
        :type s_down: frozenset
        :return: vertex left, vertex right, cond set, and the bicop
        :rtype: tuple
        """
        lv_up = len(s_down) - 1
        for (v_l, v_r, s_up), bcp in self.dct_bcp[lv_up].items():
            if ({v_l, v_r} | s_up) == ({v_down} | s_down):
                return v_l, v_r, s_up, bcp

    def _ref_count(
        self,
        tpl_first_vs: tuple[tuple[int, frozenset]] = tuple(),
        tpl_sim: tuple[int] = tuple(),
    ) -> tuple[dict, list[int]]:
        """reference counting for each vertex during simulation (quant-reg/cond-sim) workflow,
        for garbage collection (memory release)

        :param tpl_first_vs: tuple of vertices (explicitly arranged in conditioned - conditioning set)
            that are taken as known at the beginning of a simulation workflow, defaults to tuple()
        :type tpl_first_vs: tuple[tuple[int, frozenset]], optional
        :param tpl_sim: tuple of vertices in a full simulation workflow,
            gives flexibility to experienced users, defaults to tuple()
        :type tpl_sim: tuple[int], optional
        :return: reference counting for each vertex; list of source vertices in this simulation workflow from shallow to deep
        :rtype: tuple[dict, list[int]]
        """
        # * v for vertex, s for condition (frozen)set, read from right to left
        dct_first_vs = {v[0]: v for v in tpl_first_vs}
        lst_source = tpl_sim if tpl_sim else self.tpl_sim
        lst_source = [
            (dct_first_vs[v] if v in dct_first_vs else (v, frozenset(lst_source[(idx + 1) :])))
            for idx, v in enumerate(lst_source)
        ][::-1]

        # ! count in initial sim (pseudo obs that are given at the beginning of each sim path)
        dct_ref_count = {v_s_cond: 1 for v_s_cond in lst_source}

        def visit_hfunc(v_down: int, s_down: frozenset):
            v_l, v_r, s_up, _ = self._loc_bcp(v_down=v_down, s_down=s_down)
            # vertex left on upper level
            if dct_ref_count.get((v_l, s_up), 0) < 1:
                visit_hfunc(v_down=v_l, s_down=s_up)
            dct_ref_count[v_l, s_up] += 1
            # vertex right on upper level
            if dct_ref_count.get((v_r, s_up), 0) < 1:
                visit_hfunc(v_down=v_r, s_down=s_up)
            dct_ref_count[v_r, s_up] += 1
            # * vertex reached by hfunc, on lower level
            if dct_ref_count.get((v_down, s_down), 0) < 1:
                dct_ref_count[v_down, s_down] = 1
            else:
                dct_ref_count[v_down, s_down] += 1

        def visit_hinv(v_down: int, s_down: frozenset):
            v_l, v_r, s_up, _ = self._loc_bcp(v_down=v_down, s_down=s_down)
            v_up = v_r if (v_down == v_l) else v_l
            # hfunc from even upper, to generate this vertex on upper level
            if dct_ref_count.get((v_up, s_up), 0) < 1:
                visit_hfunc(v_down=v_up, s_down=s_up)
            dct_ref_count[v_up, s_up] += 1
            # vertex on lower level
            dct_ref_count[v_down, s_down] += 1
            # * vertex reached by hinv
            if dct_ref_count.get((v_down, s_up), 0) < 1:
                dct_ref_count[v_down, s_up] = 1
            else:
                dct_ref_count[v_down, s_up] += 1
            return v_down, s_up

        for v, s in lst_source:
            if len(s) < 1:
                continue
            else:
                v_next, s_next = visit_hinv(v_down=v, s_down=s)
                while len(s_next) > 0:
                    v_next, s_next = visit_hinv(v_down=v_next, s_down=s_next)

        return dct_ref_count, lst_source

    def __repr__(self) -> str:
        return pformat(
            {
                "dct_bcp": self.dct_bcp,
                "dct_tree": self.dct_tree,
                "tpl_sim": self.tpl_sim,
                "mtd_bidep": self.mtd_bidep,
            },
            indent=2,
            compact=True,
            sort_dicts=True,
            underscore_numbers=True,
        )

    def __str__(self) -> str:
        return pformat(
            object={
                "mtd_bidep": self.mtd_bidep,
                "num_dim": self.num_dim,
                "num_obs": self.num_obs,
                "num_par": self.num_par,
                "negloglik": round(self.negloglik, 4),
                "aic": round(self.aic, 4),
                "bic": round(self.bic, 4),
                "matrix": self.matrix.__str__(),
                "tpl_sim": self.tpl_sim,
            },
            compact=True,
            sort_dicts=False,
            underscore_numbers=True,
        ).replace("\\n", "")

    def draw_lv(
        self,
        lv: int = 0,
        is_bcp: bool = True,
        num_digit: int = 2,
        font_size_vertex: int = 8,
        font_size_edge: int = 7,
        f_path: Path = None,
        fig_size: tuple = None,
    ) -> tuple:
        """draw the vine structure at a given level. Each edge corresponds to a fitted bicop.
        Each node is a bicop of prev lv (is_bcp=True) or pseudo-obs of the given lv (is_bcp=False).

        :param lv: level, defaults to 0
        :type lv: int, optional
        :param is_bcp: draw the minimum spanning tree (of prev lv bicops) or pseudo observations, defaults to True
        :type is_bcp: bool, optional
        :param num_digit: number of digits to round for the weights on edges, defaults to 2
        :type num_digit: int, optional
        :param font_size_vertex: font size for vertex labels, defaults to 8
        :type font_size_vertex: int, optional
        :param font_size_edge: font size for edge labels, defaults to 7
        :type font_size_edge: int, optional
        :param f_path: file path to save the figure, defaults to None for no saving
        :type f_path: Path, optional
        :param fig_size: figure size, defaults to None
        :type fig_size: tuple, optional
        :return: fig, ax, (and file path if the figure is saved)
        :rtype: tuple
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
        ax.set_title(
            label=f"Vine Copula, Level {lv}, BiDep Metric {self.mtd_bidep}",
            fontsize=font_size_vertex + 1,
        )
        if lv == 0:
            tpl_uvw = tuple(
                (u, v, round(w, ndigits=num_digit)) for (u, v, _), w in self.dct_tree[lv].items()
            )
        elif is_bcp:
            tpl_uvw = tuple(
                (
                    self._loc_bcp(v_down=u, s_down=s_cond),
                    self._loc_bcp(v_down=v, s_down=s_cond),
                    round(w, ndigits=num_digit),
                )
                for (u, v, s_cond), w in self.dct_tree[lv].items()
            )
            tpl_uvw = tuple(
                (
                    f"{tpl_l[0]},{tpl_l[1]};"
                    + ("" if lv == 1 else "\n")
                    + f"{','.join([str(_) for _ in sorted(tpl_l[2])])}",
                    f"{tpl_r[0]},{tpl_r[1]};"
                    + ("" if lv == 1 else "\n")
                    + f"{','.join([str(_) for _ in sorted(tpl_r[2])])}",
                    w,
                )
                for (tpl_l, tpl_r, w) in tpl_uvw
            )
        else:
            tpl_uvw = tuple(
                (
                    f"{u}|{','.join([f'{_}' for _ in sorted(s_cond)])}",
                    f"{v}|{','.join([f'{_}' for _ in sorted(s_cond)])}",
                    round(w, ndigits=num_digit),
                )
                for (u, v, s_cond), w in self.dct_tree[lv].items()
            )
        G = nx.Graph()
        G.add_weighted_edges_from(tpl_uvw)
        pos = nx.planar_layout(G)
        nx.draw_networkx_nodes(
            G=G,
            pos=pos,
            ax=ax,
            node_color="white",
            node_shape="s" if (is_bcp and lv > 0) else "o",
            alpha=0.8,
            linewidths=0.5,
            edgecolors="gray",
        )
        nx.draw_networkx_labels(G=G, pos=pos, ax=ax, font_size=font_size_vertex, alpha=0.9)
        nx.draw_networkx_edges(
            G=G,
            pos=pos,
            ax=ax,
            edgelist=G.edges(),
            width=[math.log1p(0.5 + 100 * abs(tpl[2]["weight"])) for tpl in G.edges(data=True)],
            style="--",
            alpha=0.9,
        )
        nx.draw_networkx_edge_labels(
            G=G,
            pos=pos,
            ax=ax,
            edge_labels=nx.get_edge_attributes(G, "weight"),
            font_size=font_size_edge,
        )
        fig.tight_layout()
        plt.draw_if_interactive()
        if f_path:
            fig.savefig(fname=f_path, bbox_inches="tight")
            return fig, ax, G, f_path
        else:
            return fig, ax, G

    def draw_dag(
        self,
        tpl_first_vs: tuple = tuple(),
        tpl_sim: tuple = tuple(),
        title: str = "Vine Copula, Obs and BiCop",
        font_size_vertex: int = 8,
        f_path: Path = None,
        fig_size: tuple = None,
    ) -> tuple:
        """draw the directed acyclic graph (DAG) of the vine copula, with pseudo observations and bicops as nodes.
        The source nodes in simulation workflow are highlighted in yellow.

        :param tpl_first_vs: tuple of vertices (explicitly arranged in conditioned - conditioning set)
            that are taken as already simulated at the beginning of a simulation workflow,
            affecting the color of nodes, defaults to tuple()
        :type tpl_first_vs: tuple, optional
        :param tpl_sim: tuple of vertices in a full simulation workflow,
            gives flexibility to experienced users, defaults to tuple()
        :type tpl_sim: tuple, optional
        :param font_size_vertex: font size for vertex labels, defaults to 8
        :type font_size_vertex: int, optional
        :param f_path: file path to save the figure, defaults to None for no saving
        :type f_path: Path, optional
        :param fig_size: figure size, defaults to None
        :type fig_size: tuple, optional
        :return: _description_
        :rtype: tuple
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
        ax.set_title(
            label=title,
            fontsize=font_size_vertex + 1,
        )
        G = nx.DiGraph()
        pos_obs = {}
        pos_bcp = {}
        dct_label = {}
        for lv in self.dct_tree:
            # given a bcp, locate upper/lower edges and lower nodes
            lst_node_bcp = []
            lst_node_down = []
            lst_edge = []
            if lv == 0:
                # locate upper nodes only when lv==0
                lst_node_up = [(_, frozenset()) for _ in range(self.num_dim)]
                for _ in lst_node_up:
                    dct_label[_] = _[0]
                num_node = len(lst_node_up)
                loc_x = np.linspace(-num_node / 2, num_node / 2, num=num_node)
                for _ in range(num_node):
                    pos_obs[lst_node_up[_]] = loc_x[_], 1
            for (v_l, v_r, s_cond), _ in self.dct_tree[lv].items():
                # node bcp
                lst_node_bcp.append((v_l, v_r, s_cond))
                # node down
                lst_node_down.append((v_l, s_cond | {v_r}))
                lst_node_down.append((v_r, s_cond | {v_l}))
                # edge
                lst_edge.append(((v_l, s_cond), (v_l, v_r, s_cond)))
                lst_edge.append(((v_r, s_cond), (v_l, v_r, s_cond)))
                lst_edge.append(((v_l, v_r, s_cond), (v_l, s_cond | {v_r})))
                lst_edge.append(((v_l, v_r, s_cond), (v_r, s_cond | {v_l})))
            # locate lower nodes
            num_node = len(lst_node_down)
            loc_x = np.linspace(-num_node / 2, num_node / 2, num=num_node)
            for _ in range(num_node):
                pos_obs[lst_node_down[_]] = loc_x[_], -lv
            for _ in lst_node_down:
                dct_label[_] = f"{_[0]}|{','.join([f'{__}' for __ in sorted(_[1])])}"
            # locate bcp nodes
            num_node = len(lst_node_bcp)
            loc_x = np.linspace(-num_node / 2, num_node / 2, num=num_node)
            for _ in range(num_node):
                pos_bcp[lst_node_bcp[_]] = loc_x[_], -lv + 0.5
            for _ in lst_node_bcp:
                __ = "\n" * min(lv, 1)
                dct_label[_] = f"{_[0]},{_[1]};{__}{','.join([f'{__}' for __ in sorted(_[2])])}"
            G.add_edges_from(lst_edge)
        pos = pos_obs | pos_bcp
        # highlight source nodes, given tpl_first
        lst_source = self._ref_count(tpl_first_vs=tpl_first_vs, tpl_sim=tpl_sim)[1]
        # pseudo obs nodes
        lst_node = [_ for _ in G.nodes if len(_) == 2 and _ not in lst_source]
        nx.draw_networkx_nodes(
            G=G,
            pos=pos,
            ax=ax,
            nodelist=lst_node,
            node_color="white",
            node_shape="o",
            alpha=0.8,
            linewidths=0.5,
            edgecolors="gray",
        )
        nx.draw_networkx_nodes(
            G=G,
            pos=pos,
            ax=ax,
            nodelist=lst_source,
            node_color="yellow",
            node_shape="o",
            alpha=0.8,
            linewidths=0.5,
            edgecolors="gray",
        )

        # bicop nodes
        lst_node = [_ for _ in G.nodes if len(_) == 3]
        nx.draw_networkx_nodes(
            G=G,
            pos=pos,
            ax=ax,
            nodelist=lst_node,
            node_color="white",
            node_shape="s",
            alpha=0.8,
            linewidths=0.5,
            edgecolors="gray",
        )
        nx.draw_networkx_labels(
            G=G,
            pos=pos,
            ax=ax,
            labels=dct_label,
            font_size=font_size_vertex,
            font_color="black",
        )
        nx.draw_networkx_edges(
            G=G,
            pos=pos,
            ax=ax,
            edge_color="gray",
            width=0.5,
            style="--",
            alpha=0.8,
        )
        fig.tight_layout()
        plt.box(False)
        plt.draw_if_interactive()
        if f_path:
            fig.savefig(fname=f_path, bbox_inches="tight")
            return fig, ax, G, f_path
        else:
            return fig, ax, G

    def vcp_to_json(self, f_path: Path = "./vcp.json") -> Path:
        """save to a json file

        :param self: an instance of the DataVineCop dataclass
        :param f_path: file path to save the json file, defaults to './vcp.json'
        :type f_path: Path, optional
        :return: file path where the json file is saved
        :rtype: Path
        """
        f_path = Path(f_path)
        f_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_json = asdict(self)
        for lv in tmp_json["dct_bcp"]:
            tmp_json["dct_bcp"][lv] = {
                str((key[0], key[1], tuple(key[2]))): val
                for key, val in tmp_json["dct_bcp"][lv].items()
            }
            tmp_json["dct_tree"][lv] = {
                str((key[0], key[1], tuple(key[2]))): val
                for key, val in tmp_json["dct_tree"][lv].items()
            }
        with open(f_path, "w") as file:
            json.dump(
                obj=tmp_json,
                fp=file,
                indent=4,
            )
        return f_path

    def vcp_to_pth(self, f_path: Path = Path("./vcp.pth")) -> Path:
        """save to a pth file

        :param self: an instance of the DataVineCop dataclass
        :param f_path: file path to save the pth file, defaults to Path('./vcp.pth')
        :type f_path: Path, optional
        :return: file path where the pth file is saved
        :rtype: Path
        """
        f_path = Path(f_path)
        f_path.parent.mkdir(parents=True, exist_ok=True)
        with open(f_path, "wb") as file:
            torch.save(self, file)
        return f_path

    def l_pdf(self, obs_mvcp: torch.Tensor) -> torch.Tensor:
        """log probability density function (PDF) of the multivariate copula

        :param self: an instance of the DataVineCop dataclass
        :param obs_mvcp: observation of the multivariate copula, of shape (num_obs, num_dim)
        :type obs_mvcp: torch.Tensor
        :return: log probability density function (PDF) of shape (num_obs, 1), given the observation
        :rtype: torch.Tensor
        """
        # traverse all bicops in the tree, from top to bottom
        num_dim = obs_mvcp.shape[1]
        dct_obs = {_: {} for _ in range(num_dim - 1)}
        dct_obs[0] = {(idx, frozenset()): obs_mvcp[:, [idx]] for idx in range(num_dim)}
        res = torch.zeros(
            size=(obs_mvcp.shape[0], 1),
            device=obs_mvcp.device,
            dtype=obs_mvcp.dtype,
        )

        def update_obs(v: int, s_cond: frozenset):
            # * calc hfunc for pseudo obs when necessary
            # the lv of bcp
            lv = len(s_cond) - 1
            for (v_l, v_r, s_cond_bcp), bcp in self.dct_bcp[lv].items():
                # ! notice hfunc1 or hfunc2
                if v == v_l and s_cond == frozenset({v_r} | s_cond_bcp):
                    dct_obs[lv + 1][(v, s_cond)] = bcp.hfunc2(
                        obs=torch.hstack(
                            [
                                dct_obs[lv][v_l, s_cond_bcp],
                                dct_obs[lv][v_r, s_cond_bcp],
                            ]
                        )
                    )
                elif v == v_r and s_cond == frozenset({v_l} | s_cond_bcp):
                    dct_obs[lv + 1][(v, s_cond)] = bcp.hfunc1(
                        obs=torch.hstack(
                            [
                                dct_obs[lv][v_l, s_cond_bcp],
                                dct_obs[lv][v_r, s_cond_bcp],
                            ]
                        )
                    )
                else:
                    pass

        for lv in self.dct_tree:
            for (v_l, v_r, s_cond), bcp in self.dct_bcp[lv].items():
                # * update the pseudo observations
                for idx in (v_l, v_r):
                    if dct_obs[lv].get((idx, s_cond)) is None:
                        update_obs(v=idx, s_cond=s_cond)
                res += bcp.l_pdf(
                    obs=torch.hstack(
                        [
                            dct_obs[lv][(v_l, s_cond)],
                            dct_obs[lv][(v_r, s_cond)],
                        ]
                    )
                )
            if lv > 0:
                # ! garbage collection
                del dct_obs[lv - 1]

        return res

    def rosenblatt_transform(
        self,
        obs_mvcp: torch.Tensor,
        tpl_sim: tuple = tuple(),
    ) -> dict:
        """Rosenblatt transformation, from the multivariate copula (with dependence)
        to the uniform multivariate copula (independent), using constructed vine copula

        :param obs_mvcp: observation of the multivariate copula, of shape (num_obs, num_dim)
        :type obs_mvcp: torch.Tensor
        :param tpl_sim: tuple of vertices (read from right to left) in a full simulation workflow,
            gives flexibility to experienced users, defaults to tuple()
        :type tpl_sim: tuple, optional
        :return: ideally independent uniform multivariate copula, of shape (num_obs, num_dim)
        :rtype: dict
        """
        num_dim = self.num_dim
        if not tpl_sim:
            tpl_sim = self.tpl_sim
        tpl_sim_v_s_cond = tuple(
            (v, frozenset(tpl_sim[idx + 1 :])) for idx, v in enumerate(tpl_sim)
        )
        dct_obs = {_: {} for _ in range(num_dim)}
        dct_obs[0] = {(idx, frozenset()): obs_mvcp[:, [idx]] for idx in range(num_dim)}

        def update_obs(v: int, s_cond: frozenset):
            # * calc hfunc for pseudo obs when necessary
            # the lv of bcp
            lv = len(s_cond) - 1
            for (v_l, v_r, s_cond_bcp), bcp in self.dct_bcp[lv].items():
                # ! notice hfunc1 or hfunc2
                if v == v_l and s_cond == frozenset({v_r} | s_cond_bcp):
                    dct_obs[lv + 1][(v, s_cond)] = bcp.hfunc2(
                        obs=torch.hstack(
                            [
                                dct_obs[lv][v_l, s_cond_bcp],
                                dct_obs[lv][v_r, s_cond_bcp],
                            ]
                        )
                    )
                elif v == v_r and s_cond == frozenset({v_l} | s_cond_bcp):
                    dct_obs[lv + 1][(v, s_cond)] = bcp.hfunc1(
                        obs=torch.hstack(
                            [
                                dct_obs[lv][v_l, s_cond_bcp],
                                dct_obs[lv][v_r, s_cond_bcp],
                            ]
                        )
                    )
                else:
                    pass

        for lv in self.dct_tree:
            for (v_l, v_r, s_cond), _ in self.dct_bcp[lv].items():
                # * update the pseudo observations
                for idx in (v_l, v_r):
                    if dct_obs[lv].get((idx, s_cond)) is None:
                        update_obs(v=idx, s_cond=s_cond)
                if (v_l, s_cond | {v_r}) in tpl_sim_v_s_cond:
                    update_obs(v=v_l, s_cond=s_cond | {v_r})
                if (v_r, s_cond | {v_l}) in tpl_sim_v_s_cond:
                    update_obs(v=v_r, s_cond=s_cond | {v_l})
            if lv > 0:
                # ! garbage collection
                for v_s_cond in dict(dct_obs[lv - 1]):
                    if v_s_cond not in tpl_sim_v_s_cond:
                        del dct_obs[lv - 1][v_s_cond]
        dct_obs = {
            k: v
            for dct_lv in dct_obs.values()
            for k, v in dct_lv.items()
            if (k in tpl_sim_v_s_cond)
        }
        return dct_obs

    def sim(
        self,
        num_sim: int = 1,
        dct_first_vs: dict = {},
        tpl_sim: tuple = tuple(),
        seed: int = 0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """full simulation/ quantile-regression/ conditional-simulation using the vine copula.
        Sequentially for each beginning vertex in the tpl_sim
        (from right to left, as from shallower lv to deeper lv in the DAG),
        walk upward by calling hinv until the top vertex (whose cond set is empty) is reached.
        (Recursively) call hfunc for the other upper vertex if necessary.

        :param num_sim: number of simulations; ignored when dct_first_vs is not empty
        :type num_sim: int
        :param dct_first_vs: dict of {(vertex,cond_set): torch.Tensor(size=(n,1))}
            in quantile regression/ conditional simulation, where vertices are taken as given already; defaults to {}
        :type dct_first_vs: dict, optional
        :param tpl_sim: tuple of vertices (read from right to left) in a full simulation workflow,
            gives flexibility to experienced users, defaults to tuple()
        :type tpl_sim: tuple, optional
        :param seed: random seed for torch.manual_seed(), defaults to 0
        :type seed: int, optional
        :param device: device for torch.rand(), defaults to 'cpu'
        :type device: str, optional
        :param dtype: dtype for torch.rand(), defaults to torch.float64
        :type dtype: torch.dtype, optional
        :return: simulated observations of the vine copula, of shape (num_sim, num_dim)
        :rtype: torch.Tensor
        """
        dct_obs = dct_first_vs.copy()
        # * source vertices in each path; reference counting for whole DAG
        dct_ref_count, lst_source = self._ref_count(
            tpl_first_vs=tuple(dct_first_vs), tpl_sim=tpl_sim
        )

        def _update_ref_count(v: int, s: frozenset) -> None:
            # * countdown and release memory if necessary
            dct_ref_count[v, s] -= 1
            if dct_ref_count[v, s] < 1:
                del dct_obs[v, s]

        def visit_hfunc(v_down: int, s_down: frozenset) -> None:
            """
            hfunc from (v_l,s_up) and (v_r,s_up) (both may not exist yet and recursively call visit_hfunc if necessary)
            to (v_down,s_down); then update dct_obs and dct_ref_count and do garbage collection
            """
            # * locate the bicop on upper level that connects the 3 vertices
            v_l, v_r, s_up, bcp = self._loc_bcp(v_down=v_down, s_down=s_down)
            # ! hfunc from even upper, to visit this vertex on upper level
            for v in (v_l, v_r):
                if dct_obs.get((v, s_up), None) is None:
                    visit_hfunc(v_down=v, s_down=s_up)
            dct_obs[v_down, s_down] = (bcp.hfunc1 if v_down == v_r else bcp.hfunc2)(
                torch.hstack([dct_obs[v_l, s_up], dct_obs[v_r, s_up]])
            )
            # * garbage collection check
            for v, s in (
                (v_l, s_up),
                (v_r, s_up),
                (v_down, s_down),
            ):
                _update_ref_count(v=v, s=s)

        def visit_hinv(v_down: int, s_down: frozenset) -> tuple:
            """
            hinv from (v_down,s_down) (surely exist) and (v_up,s_up)
            (may not exist yet and recursively call visit_hfunc if necessary)
            to (v_down,s_up); then update dct_obs and dct_ref_count and do garbage collection
            """
            # * locate the bicop on upper level that connects the 3 vertices
            v_l, v_r, s_up, bcp = self._loc_bcp(v_down=v_down, s_down=s_down)
            # ! if v_down==v_r then go hinv1([(v_l,s_up), (v_r,s_down)])
            is_down_right = v_down == v_r
            (v_up, hinv) = (v_l, bcp.hinv1) if is_down_right else (v_r, bcp.hinv2)
            # ! hfunc from even upper, to visit this vertex on upper level
            if dct_obs.get((v_up, s_up), None) is None:
                visit_hfunc(v_down=v_up, s_down=s_up)
            dct_obs[v_down, s_up] = hinv(
                torch.hstack(
                    [
                        dct_obs[v_up, s_up],
                        dct_obs[v_down, s_down],
                    ]
                    if is_down_right
                    else [
                        dct_obs[v_down, s_down],
                        dct_obs[v_up, s_up],
                    ]
                )
            )
            # * garbage collection check
            for v, s in (
                (v_up, s_up),
                (v_down, s_down),
                (v_down, s_up),
            ):
                _update_ref_count(v=v, s=s)
            # * return the next vertex
            return v_down, s_up

        torch.manual_seed(seed=seed)
        # * init sim of U_mvcp (multivariate independent copula)
        dim_sim = self.num_dim - len(dct_first_vs)
        if dim_sim > 0:
            # ! skip for quant-reg
            U_mvcp = torch.rand(size=(num_sim, dim_sim), device=device, dtype=dtype)
        # * update dct_obs and dct_ref_count
        idx = 0
        for v, s in lst_source:
            if (v, s) not in dct_obs:
                dct_obs[v, s] = U_mvcp[:, [idx]]
                idx += 1
            # ! let the top level obs (target vertices) escape garbage collection
            dct_ref_count[v, frozenset()] += 1
            # update ref count
            dct_ref_count[v, s] -= 1
        del seed, dct_first_vs, idx
        if dim_sim > 0:
            del U_mvcp
        for v, s in lst_source:
            # walk the path if cond set is not empty
            if len(s):
                # call hinv and update vertex/cond-set iteratively to walk towards target vertex (top lv)
                v_next, s_next = visit_hinv(v_down=v, s_down=s)
                while len(s_next):
                    v_next, s_next = visit_hinv(v_down=v_next, s_down=s_next)
        # * sort pseudo obs by key
        return torch.hstack([val for _, val in sorted(dct_obs.items(), key=itemgetter(0))])

    def cdf(
        self,
        obs_mvcp: torch.Tensor,
        num_sim: int = 10000,
        seed: int = 0,
    ) -> torch.Tensor:
        """cumulative distribution function (CDF) of the multivariate copula given observations, by Monte Carlo.

        :param self: an instance of the DataVineCop dataclass
        :param obs_mvcp: observations of the multivariate copula, of shape (num_obs, num_dim)
        :type obs_mvcp: torch.Tensor
        :param num_sim: number of simulations, defaults to 10000
        :type num_sim: int, optional
        :param seed: random seed for torch.manual_seed(), defaults to 0
        :type seed: int, optional
        :return: cumulative distribution function (CDF) of shape (num_obs, 1), given observations
        :rtype: torch.Tensor
        """
        # * both obs_mvcp and obs_sim are of 2 dimensions
        obs_sim = self.sim(
            num_sim=num_sim, seed=seed, device=obs_mvcp.device, dtype=obs_mvcp.dtype
        )
        # * unsqueeze for broadcasting (num_sim, obs_mvcp.shape[0], num_dim) -> (obs_mvcp.shape[0], 1)
        return (obs_sim.unsqueeze(dim=1) <= obs_mvcp).all(
            dim=2,
            keepdim=True,
        ).sum(
            axis=0,
            keepdim=False,
        ) / obs_sim.shape[0]
