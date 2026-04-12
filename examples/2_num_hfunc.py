"""Sampling-order h-function cost example script."""

import matplotlib.pyplot as plt
import pandas as pd
import torch

import torchvinecopulib as tvc

try:
    from examples._example_utils import configure_matplotlib, export_figure
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from _example_utils import configure_matplotlib, export_figure


def worst_sample_order(mdl_vcp: tvc.VineCop) -> list[int]:
    """Greedily choose the currently most expensive next vertex."""
    last_tree_vertex = set(range(mdl_vcp.num_dim)) - set(mdl_vcp.first_tree_vertex)
    sample_order: list[int] = []
    for v_s_parent in mdl_vcp.struct_obs[::-1]:
        cost_best = -float("inf")
        cand_v: set[int] = set()
        for v_s, cond_ed in v_s_parent.items():
            if cond_ed:
                v_l, v_r = map(int, cond_ed.split(","))
                if v_l not in sample_order and v_r not in sample_order:
                    cand_v.update((v_l, v_r))
            elif v_s[0] not in sample_order:
                cand_v.add(v_s[0])
        cand_v_last = cand_v & last_tree_vertex
        if cand_v_last:
            cand_v = cand_v_last
        for v in sorted(cand_v):
            _, _, cost = mdl_vcp.ref_count_hfunc(
                num_dim=mdl_vcp.num_dim,
                struct_obs=mdl_vcp.struct_obs,
                sample_order=sample_order + [v],
            )
            if cost > cost_best:
                cost_best = cost
                v_best = v
        sample_order.append(v_best)
    return sample_order


def main() -> dict:
    """Compare best and worst sampling-order costs and export the docs figure."""
    configure_matplotlib()
    torch.manual_seed(42)
    device = "cpu"
    dtype = torch.float64
    is_cop_scale = True

    rows: list[dict[str, int]] = []
    for num_dim in range(2, 34, 4):
        for vine_kind in ("dvine", "cvine"):
            mdl_vcp = tvc.VineCop(num_dim=num_dim, is_cop_scale=is_cop_scale).to(device)
            mdl_vcp.fit(
                obs=torch.rand(size=(10, num_dim), dtype=dtype, device=device),
                mtd_vine=vine_kind,
                thresh_trunc=-1,
            )
            worst = worst_sample_order(mdl_vcp)
            best_cost = mdl_vcp.ref_count_hfunc(
                num_dim=mdl_vcp.num_dim,
                struct_obs=mdl_vcp.struct_obs,
                sample_order=mdl_vcp.sample_order,
            )[2]
            worst_cost = mdl_vcp.ref_count_hfunc(
                num_dim=mdl_vcp.num_dim,
                struct_obs=mdl_vcp.struct_obs,
                sample_order=worst,
            )[2]
            rows.append(
                {
                    "num_dim": num_dim,
                    "vine": vine_kind,
                    "best": best_cost,
                    "worst": worst_cost,
                }
            )

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 1, figsize=(8.4, 5.2), constrained_layout=True)
    color_map = {"dvine": "tab:blue", "cvine": "tab:orange"}
    line_styles = {"best": "--", "worst": ":"}
    marker_map = {"best": "o", "worst": "s"}
    for vine_kind in ("dvine", "cvine"):
        sub = df[df["vine"] == vine_kind].sort_values("num_dim")
        for case in ("best", "worst"):
            ax.plot(
                sub["num_dim"],
                sub[case],
                color=color_map[vine_kind],
                linestyle=line_styles[case],
                marker=marker_map[case],
                linewidth=1.5,
                label=f"{vine_kind} {case}",
            )
    ax.set_xlabel("Number of dimensions")
    ax.set_ylabel("h-function calls")
    ax.grid(True, linewidth=0.5, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), ncol=2, fontsize="small")

    path = export_figure(fig, "num_hfunc.svg")
    plt.close(fig)
    return {
        "figure_path": str(path),
        "rows": len(df),
        "max_hfunc_calls": int(df[["best", "worst"]].to_numpy().max()),
    }


if __name__ == "__main__":
    print(main())
