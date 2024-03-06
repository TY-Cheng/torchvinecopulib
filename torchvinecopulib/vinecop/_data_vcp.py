import math
import pickle
from abc import ABC
from dataclasses import dataclass
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
    """bivariate dependency measures of edges in trees, stored as {level: {(vertex_left, vertex_right, frozenset_cond): float}}"""
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
    def diag(self) -> list:
        """diagonal elements in the structure matrix

        :param self: an instance of the DataVineCop dataclass
        :return: list of diagonal elements
        :rtype: list
        """
        lst_diag = []
        for lv in sorted(self.dct_tree, reverse=True):
            v_diag = None
            for i_lv in range(lv, -1, -1):
                for v_l, v_r, s_and in self.dct_tree[i_lv]:
                    if (
                        (v_diag is None)
                        and (v_l not in lst_diag)
                        and (v_r not in lst_diag)
                    ):
                        # ! pick the node with smaller index (v_l < v_r), then mat <-> structure is bijection
                        v_diag = v_l
                        lst_diag.append(v_diag)
                        if lv == 0:
                            lst_diag.append(v_r)

        return lst_diag

    @property
    def matrix(self) -> np.array:
        """structure matrix: upper triangular, in row-major order (vertices of the same level are in the same row)

        :param self: an instance of the DataVineCop dataclass
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
                for v_l, v_r, s_and in self.dct_tree[i_lv]:
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

        :param self: an instance of the DataVineCop dataclass
        :return: negative log likelihood
        :rtype: float
        """
        return sum(
            [
                bcp.negloglik
                for dct_lv in self.dct_bcp.values()
                for bcp in dct_lv.values()
            ]
        )

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
        return sum(
            [bcp.num_par for dct_lv in self.dct_bcp.values() for bcp in dct_lv.values()]
        )

    def _loc_bcp(self, v_down: int, s_down: frozenset) -> tuple:
        # * locate the bicop on upper level that generates this pseudo obs
        lv_up = len(s_down) - 1
        for (v_l, v_r, s_up), bcp in self.dct_bcp[lv_up].items():
            if ({v_l, v_r} | s_up) == ({v_down} | s_down):
                return v_l, v_r, s_up, bcp

    @property
    def ref_count(self) -> dict:
        """reference counting for each vertex during simulation workflow, for garbage collection

        :param self: an instance of the DataVineCop dataclass
        :return: reference counting for each vertex
        :rtype: dict
        """
        # * v for vertex, s for condition set
        lst_diag = self.diag
        lst_diag = [
            (v, frozenset(lst_diag[(idx + 1) :])) for idx, v in enumerate(lst_diag)
        ][::-1]
        # count in initial sim
        dct_ref_count = {(v, s): 1 for v, s in lst_diag}

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

        for v_diag, s in lst_diag:
            if len(s) < 1:
                continue
            else:
                v_next, s_next = visit_hinv(v_down=v_diag, s_down=s)
                while len(s_next) > 0:
                    v_next, s_next = visit_hinv(v_down=v_next, s_down=s_next)

        return dct_ref_count

    def __repr__(self) -> str:
        return pformat(
            {
                "mtd_bidep": self.mtd_bidep,
                "dct_tree": self.dct_tree,
                "dct_bcp": self.dct_bcp,
            },
            indent=2,
            compact=True,
            sort_dicts=True,
            underscore_numbers=True,
        )

    def __str__(self) -> str:
        return pformat(
            object={
                "num_dim": self.num_dim,
                "num_obs": self.num_obs,
                "num_par": self.num_par,
                "negloglik": round(self.negloglik, 4),
                "aic": round(self.aic, 4),
                "bic": round(self.bic, 4),
                "matrix": self.matrix.__str__(),
            },
            compact=True,
            sort_dicts=False,
            underscore_numbers=True,
        ).replace("\\n", "")

    def draw(
        self,
        lv: int = 0,
        is_mst: bool = True,
        ndigits: int = 2,
        font_size_vertex: int = 8,
        font_size_edge: int = 7,
        f_path: Path = None,
        figsize: tuple = None,
    ) -> Path:
        """draw the vine structure of the given level.

        :param self: an instance of the DataVineCop dataclass
        :param lv: level, defaults to 0
        :type lv: int, optional
        :param is_mst: draw the minimum spanning tree (of bicops) or pseudo observations, defaults to True
        :type is_mst: bool, optional
        :param ndigits: number of digits to round for the weights on edges, defaults to 2
        :type ndigits: int, optional
        :param font_size_vertex: font size for vertex labels, defaults to 8
        :type font_size_vertex: int, optional
        :param font_size_edge: font size for edge labels, defaults to 7
        :type font_size_edge: int, optional
        :param f_path: file path to save the figure, defaults to None
        :type f_path: Path, optional
        :param figsize: figure size, defaults to None
        :type figsize: tuple, optional
        :return: file path where the figure is saved
        :rtype: Path
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        if f_path is None:
            f_path = Path("./vcp.png")
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        ax.set_title(label=f"Vine Copula, Level {lv}", fontsize=font_size_vertex + 1)
        if lv == 0:
            tpl_uvw = tuple(
                (u, v, round(w, ndigits=ndigits))
                for (u, v, s_and), w in self.dct_tree[lv].items()
            )
        elif is_mst:
            tpl_uvw = tuple(
                (
                    self._loc_bcp(v_down=u, s_down=s_and),
                    self._loc_bcp(v_down=v, s_down=s_and),
                    round(w, ndigits=ndigits),
                )
                for (u, v, s_and), w in self.dct_tree[lv].items()
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
                    f"{u}|{','.join([f'{_}' for _ in sorted(s_and)])}",
                    f"{v}|{','.join([f'{_}' for _ in sorted(s_and)])}",
                    round(w, ndigits=ndigits),
                )
                for (u, v, s_and), w in self.dct_tree[lv].items()
            )
        G = nx.Graph()
        G.add_weighted_edges_from(tpl_uvw)
        pos = nx.planar_layout(G)
        nx.draw_networkx_nodes(
            G=G,
            pos=pos,
            ax=ax,
            node_color="white",
            node_shape="s" if is_mst else "o",
            alpha=0.8,
            linewidths=0.5,
            edgecolors="gray",
        )
        nx.draw_networkx_labels(
            G=G, pos=pos, ax=ax, font_size=font_size_vertex, alpha=0.9
        )
        nx.draw_networkx_edges(
            G=G,
            pos=pos,
            ax=ax,
            edgelist=G.edges(),
            width=[
                math.log1p(0.5 + 100 * abs(tpl[2]["weight"]))
                for tpl in G.edges(data=True)
            ],
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
        #
        plt.show()
        fig.savefig(fname=f_path, bbox_inches="tight")
        return f_path, fig, ax

    def to_json(self, f_path: Path = "./vcp.json") -> Path:
        """save to a json file

        :param self: an instance of the DataVineCop dataclass
        :param f_path: file path to save the json file, defaults to './vcp.json'
        :type f_path: Path, optional
        :return: file path where the json file is saved
        :rtype: Path
        """
        f_path = Path(f_path)
        f_path.parent.mkdir(parents=True, exist_ok=True)
        with open(f_path, "w") as file:
            file.write(self.__repr__())
        return f_path

    def to_pkl(self, f_path: Path = Path("./vcp.pkl")) -> Path:
        """save to a pickle file

        :param self: an instance of the DataVineCop dataclass
        :param f_path: file path to save the pickle file, defaults to Path('./vcp.pkl')
        :type f_path: Path, optional
        :return: file path where the pickle file is saved
        :rtype: Path
        """
        f_path = Path(f_path)
        f_path.parent.mkdir(parents=True, exist_ok=True)
        with open(f_path, "wb") as file:
            pickle.dump(self, file)
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

        def update_obs(idx: int, s_cond: frozenset):
            # * calc hfunc for pseudo obs when necessary
            lv = len(s_cond) - 1
            for (v_l, v_r, s_and), bcp in self.dct_bcp[lv].items():
                # ! notice hfunc1 or hfunc2
                if idx == v_l and s_cond == frozenset({v_r} | s_and):
                    dct_obs[lv + 1][(idx, s_cond)] = bcp.hfunc2(
                        obs=torch.hstack(
                            [
                                dct_obs[lv][v_l, s_and],
                                dct_obs[lv][v_r, s_and],
                            ]
                        )
                    )
                elif idx == v_r and s_cond == frozenset({v_l} | s_and):
                    dct_obs[lv + 1][(idx, s_cond)] = bcp.hfunc1(
                        obs=torch.hstack(
                            [
                                dct_obs[lv][v_l, s_and],
                                dct_obs[lv][v_r, s_and],
                            ]
                        )
                    )
                else:
                    pass

        for lv in self.dct_tree:
            for (v_l, v_r, s_and), bcp in self.dct_bcp[lv].items():
                # * update the pseudo observations
                for idx in (v_l, v_r):
                    if dct_obs[lv].get((idx, s_and)) is None:
                        update_obs(idx=idx, s_cond=s_and)
                obs_l = dct_obs[lv][(v_l, s_and)]
                obs_r = dct_obs[lv][(v_r, s_and)]
                res += bcp.l_pdf(obs=torch.hstack([obs_l, obs_r]))
            if lv > 0:
                # ! garbage collection
                del dct_obs[lv - 1]

        return res

    def sim(
        self,
        num_sim: int,
        seed: int = 0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """simulate from the vine copula, as scheduled task with reference counting for garbage collection.
        Sequentially for each beginning vertex in the diagonal vertices,
        move upward by calling hinv until the top vertex (whose cond set is empty) is reached.
        (Recursively) call hfunc for the other upper vertex if necessary.

        :param self: an instance of the DataVineCop dataclass
        :param num_sim: number of simulations
        :type num_sim: int
        :param seed: random seed for torch.manual_seed(), defaults to 0
        :type seed: int, optional
        :param device: device for torch.rand(), defaults to 'cpu'
        :type device: str, optional
        :param dtype: dtype for torch.rand(), defaults to torch.float64
        :type dtype: torch.dtype, optional
        :return: simulated observations of the vine copula, of shape (num_sim, num_dim)
        :rtype: torch.Tensor
        """
        torch.manual_seed(seed=seed)
        num_dim = self.num_dim
        dct_ref_count = self.ref_count
        # * starting vertices in each path
        lst_diag = self.diag
        lst_diag = [
            (v, frozenset(lst_diag[(idx + 1) :])) for idx, v in enumerate(lst_diag)
        ][::-1]
        # * initial sim of U_mvcp (independent uniform multivariate copula)
        dct_obs = torch.rand(size=(num_sim, num_dim), device=device, dtype=dtype)
        # * prepare dct_obs and update dct_ref_count
        dct_obs = {v_s: dct_obs[:, [idx]] for idx, v_s in enumerate(lst_diag)}
        for v_s in lst_diag:
            dct_ref_count[v_s] -= 1
        # ! let the top level obs skip garbage collection
        for idx in range(num_dim):
            dct_ref_count[idx, frozenset()] += 1

        def _update_ref_count(v: int, s: frozenset) -> None:
            # * countdown and release memory if necessary
            dct_ref_count[v, s] -= 1
            if dct_ref_count[v, s] < 1:
                del dct_obs[v, s]

        def visit_hfunc(v_down: int, s_down: frozenset) -> None:
            """
            hfunc from (v_l,s_up) and (v_r,s_up) (both may not exist yet)
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
            hinv from (v_down,s_down) (surely exist) and (v_up,s_up) (may not exist yet)
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

        for v_diag, s in lst_diag:
            if len(s):
                v_next, s_next = visit_hinv(v_down=v_diag, s_down=s)
                while len(s_next):
                    v_next, s_next = visit_hinv(v_down=v_next, s_down=s_next)
        # * sort pseudo obs by key
        return torch.hstack([v for _, v in sorted(dct_obs.items(), key=itemgetter(0))])

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
        return (obs_sim.unsqueeze_(dim=1) <= obs_mvcp).all(
            dim=2,
            keepdim=True,
        ).sum(
            axis=0,
            keepdim=False,
        ) / obs_sim.shape[0]
