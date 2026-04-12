from __future__ import annotations

import math
from pathlib import Path

import torch

__all__ = ["VineCopPlotMixin"]


class VineCopPlotMixin:
    @torch.no_grad()
    def draw_lv(
        self,
        lv: int = 0,
        is_bcp: bool = True,
        title: str | None = None,
        num_digit: int = 2,
        font_size_vertex: int = 8,
        font_size_edge: int = 7,
        f_path: Path = None,
        fig_size: tuple = None,
    ) -> tuple:
        """Draw the weighted undirected graph at a single level of the vine copula.

        This constructs a NetworkX graph of bivariate-copula edges at level `lv`, where nodes
        represent either raw variables (`lv=0`), parent-copula modules, or pseudo-observations.
        Edge widths encode dependence strength.

        Args:
            lv (int, optional): Level to draw. Defaults to 0.
            is_bcp (bool, optional): If True, nodes are parent‐bicop "l,r;s". Otherwise, nodes are pseudo‐obs "v|s". Defaults to True.
            title (str | None, optional): Title of the plot. Defaults to ``f"Vine level {lv}"``.
            num_digit (int, optional): Number of decimal digits for edge weights. Defaults to 2.
            font_size_vertex (int, optional): Font size for vertex labels. Defaults to 8.
            font_size_edge (int, optional): Font size for edge labels. Defaults to 7.
            f_path (Path, optional): Path to save the figure. Defaults to None.
            fig_size (tuple, optional): Figure size. Defaults to None.

        Raises:
            ImportError: If matplotlib or networkx is not installed.

        Returns:
            tuple: Figure, axis, graph object, and file path (if saved).
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "Please install matplotlib and networkx to draw the vine copula."
            ) from e
        tree = self.tree_bidep[lv]
        edge_weight = []
        if lv == 0:
            # * level‐0: plain variable indices
            for (u, v, *_), w in tree.items():
                edge_weight.append((u, v, round(w.item(), num_digit)))
        elif is_bcp:
            # ! nodes are the parent‐bicop "l,r;s"
            for (u, v, *s), w in tree.items():
                # * parent bicop cond_ed str
                label_u = self.struct_obs[lv][(u, *s)]
                label_v = self.struct_obs[lv][(v, *s)]
                # * append the cond_ing set
                sep = "" if lv == 1 else "\n"
                label_u = f"""{label_u};{sep}{
                    ",".join(str(x) for x in sorted(self.struct_bcp[label_u]["cond_ing"]))
                }"""
                label_v = f"""{label_v};{sep}{
                    ",".join(str(x) for x in sorted(self.struct_bcp[label_v]["cond_ing"]))
                }"""
                edge_weight.append((label_u, label_v, round(w.item(), num_digit)))
        else:
            # ! nodes are pseudo‐obs "v|s"
            for (u, v, *s), w in tree.items():
                s_str = ",".join(str(x) for x in sorted(s))
                label_u = f"{u}|{s_str}"
                label_v = f"{v}|{s_str}"
                edge_weight.append((label_u, label_v, round(w.item(), num_digit)))

        # * weighted undirected graph
        G = nx.Graph()
        G.add_weighted_edges_from(edge_weight)
        fig, ax = plt.subplots(figsize=fig_size)
        if title is None:
            title = f"Vine level {lv}"
        ax.set_title(title, fontsize=font_size_vertex + 1)
        pos = nx.planar_layout(G)
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color="white",
            node_shape="s" if (is_bcp and lv > 0) else "o",
            linewidths=0.5,
            edgecolors="gray",
            alpha=0.8,
        )
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size_vertex, font_color="black")
        # * scale line‐width by weight
        widths = [math.log1p(0.5 + 100 * abs(data["weight"])) for _, _, data in G.edges(data=True)]
        nx.draw_networkx_edges(G, pos, ax=ax, width=widths, style="--", alpha=0.9)
        nx.draw_networkx_edge_labels(
            G,
            pos,
            ax=ax,
            edge_labels=nx.get_edge_attributes(G, "weight"),
            font_size=font_size_edge,
        )
        ax.set_axis_off()
        fig.tight_layout()
        plt.draw_if_interactive()

        if f_path:
            fig.savefig(f_path, bbox_inches="tight")
            return fig, ax, G, f_path
        return fig, ax, G

    @torch.no_grad()
    def draw_dag(
        self,
        sample_order: tuple[int, ...] = None,
        title: str = "Vine comp graph",
        font_size_vertex: int = 8,
        f_path: Path = None,
        fig_size: tuple = None,
    ) -> tuple:
        """Draw the computational graph (DAG) of the vine copula.

        This creates a directed graph where edges flow from upstream pseudo-observations and
        pair-copula modules to downstream pseudo-observations, laid out by vine level.

        Args:
            sample_order (tuple[int, ...], optional): Variable sampling order. Defaults to `self.sample_order`.
            title (str, optional): Title of the plot. Defaults to "Vine comp graph".
            font_size_vertex (int, optional): Font size for vertex labels. Defaults to 8.
            f_path (Path, optional): Path to save the figure. If provided, the figure will be saved. Defaults to None.
            fig_size (tuple, optional): Figure size. Defaults to None.

        Raises:
            ImportError: If matplotlib or networkx is not installed.

        Returns:
            tuple: Figure, axis, graph object, and file path (if saved).
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "Please install matplotlib and networkx to draw the vine copula."
            ) from e

        G = nx.DiGraph()
        labels: dict = {}
        pos_obs: dict = {}
        pos_bcp: dict = {}

        def add_level(lv: int):
            edges = []
            bicops = []
            downstream = []
            # * lv-0 marginals
            if lv == 0:
                xs = np.linspace(-self.num_dim / 2, self.num_dim / 2, self.num_dim)
                for v, x in enumerate(xs):
                    node = (v, frozenset())
                    labels[node] = str(v)
                    pos_obs[node] = (float(x), 1.0)
            # * traverse around the bcp; sorted!
            for v_l, v_r, *cond in sorted(self.tree_bidep[lv]):
                cond_set = frozenset(cond)
                bcp = (v_l, v_r, cond_set)
                bicops.append(bcp)
                up_l = (v_l, cond_set)
                up_r = (v_r, cond_set)
                down_l = (v_l, cond_set | {v_r})
                down_r = (v_r, cond_set | {v_l})
                downstream.extend([down_l, down_r])
                # * edges: upstream → bicop → downstream
                edges += [
                    (up_l, bcp),
                    (up_r, bcp),
                    (bcp, down_l),
                    (bcp, down_r),
                ]
                # * labels
                labels[down_l] = f"{down_l[0]}|{','.join(map(str, sorted(down_l[1])))}"
                labels[down_r] = f"{down_r[0]}|{','.join(map(str, sorted(down_r[1])))}"
                br = "\n" if lv > 0 else ""
                labels[bcp] = f"{v_l},{v_r};{br}{','.join(map(str, sorted(cond_set)))}"
            # * layout downstream at y = –lv
            if downstream:
                xs = np.linspace(-len(downstream) / 2, len(downstream) / 2, len(downstream))
                for i, node in enumerate(downstream):
                    pos_obs[node] = (float(xs[i]), float(-lv))
            # * layout bicops at y = –lv + 0.5
            if bicops:
                xs = np.linspace(-len(bicops) / 2, len(bicops) / 2, len(bicops))
                for i, node in enumerate(bicops):
                    pos_bcp[node] = (float(xs[i]), float(-lv + 0.5))
            return edges

        # * accumulate over all levels
        all_edges = []
        for lv in range(len(self.tree_bidep)):
            all_edges.extend(add_level(lv))
        G.add_edges_from(all_edges)
        # * layout dictionary
        pos = {**pos_obs, **pos_bcp}
        # * pseudo-obs to highlight
        _, node_source, _ = self.ref_count_hfunc(
            num_dim=self.num_dim,
            struct_obs=self.struct_obs,
            sample_order=sample_order if sample_order is not None else self.sample_order,
        )
        node_source = [(v_s[0], frozenset(v_s[1:])) for v_s in node_source]
        # * draw
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
        ax.set_title(label=title, fontsize=font_size_vertex + 1)
        # * pseudo-obs (white)
        node_obs = [_ for _ in G.nodes if len(_) == 2 and _ not in node_source]
        nx.draw_networkx_nodes(
            G=G,
            pos=pos,
            nodelist=node_obs,
            ax=ax,
            node_shape="o",
            node_color="white",
            edgecolors="gray",
            linewidths=0.5,
            alpha=0.8,
        )
        # * pseudo-obs (yellow)
        nx.draw_networkx_nodes(
            G=G,
            pos=pos,
            nodelist=node_source,
            ax=ax,
            node_shape="o",
            node_color="yellow",
            edgecolors="gray",
            linewidths=0.5,
            alpha=0.9,
        )
        # * bcp nodes
        node_bcp = [_ for _ in G.nodes if len(_) == 3]
        nx.draw_networkx_nodes(
            G=G,
            pos=pos,
            nodelist=node_bcp,
            ax=ax,
            node_shape="s",
            node_color="white",
            edgecolors="gray",
            linewidths=0.5,
            alpha=0.8,
        )
        nx.draw_networkx_labels(
            G=G,
            pos=pos,
            labels=labels,
            ax=ax,
            font_size=font_size_vertex,
            font_color="black",
        )
        nx.draw_networkx_edges(
            G=G,
            pos=pos,
            ax=ax,
            edge_color="gray",
            style="--",
            width=0.5,
            alpha=0.8,
        )
        ax.set_axis_off()
        fig.tight_layout()
        plt.draw_if_interactive()
        if f_path:
            fig.savefig(fname=f_path, bbox_inches="tight")
            return fig, ax, G, f_path
        else:
            return fig, ax, G
