import os
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import ScalarFormatter, NullFormatter



def load_data(file_path):
    if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    return df


def make_binned_heatmaps(
    file_path,
    zlims=None,
    cfd_prediction=None,
):
    df = load_data(file_path)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 16

    if zlims is None:
        zlims = {
            "LCOW": (0.65, 1.15, 10),
            "SEC": (3.5, 6.5, 10),
        }

    # Only plot LCOW and SEC, one plot with two subplots
    metrics = [
        ("SEC", "Specific energy consumption (kWh/m³)", "viridis"),
        ("LCOW", "Levelized cost of water ($/m³)", "RdYlBu_r"),
    ]

    n_metrics = len(metrics)
    n_rows = 1
    n_cols = n_metrics

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), sharex=False, sharey=False
    )
    axes = axes.flatten() if n_metrics > 1 else [axes]

    subset = df.copy()

    sh_vals = np.unique(np.sort(subset["Sherwood number"].values))
    pn_vals = np.unique(np.sort(subset["Power number"].values))

    if len(sh_vals) < 2 or len(pn_vals) < 2:
        raise ValueError("Not enough unique Sh or Pn values to form bins.")

    sh_edges = np.geomspace(sh_vals[0], sh_vals[-1], len(sh_vals) + 1)
    pn_edges = np.geomspace(pn_vals[0], pn_vals[-1], len(pn_vals) + 1)

    x = subset["Sherwood number"].values
    y = subset["Power number"].values

    for ax, (metric, label, cmap_name) in zip(axes, metrics):
        z = subset[metric].copy()

        H, _, _ = np.histogram2d(x, y, bins=[sh_edges, pn_edges], weights=z)
        counts, _, _ = np.histogram2d(x, y, bins=[sh_edges, pn_edges])
        Zbinned = np.full_like(H, np.nan, dtype=np.float64)
        valid = counts != 0
        Zbinned[valid] = H[valid] / counts[valid]

        X, Y = np.meshgrid(sh_edges, pn_edges, indexing="ij")

        zmin_print, zmax_print = z.min(), z.max()
        print(f"Zmin: {zmin_print}, Zmax: {zmax_print} for metric: {metric}")
        zmin = zlims.get(metric)[0]
        zmax = zlims.get(metric)[1]
        n_levels = zlims.get(metric)[2]
        levels = np.linspace(zmin, zmax, n_levels + 1)
        n_levels = len(levels) - 1
        base_cmap = plt.get_cmap(cmap_name)
        discrete_cmap = ListedColormap(base_cmap(np.linspace(0, 1, n_levels)))
        norm = BoundaryNorm(boundaries=levels, ncolors=n_levels)

        pc = ax.pcolormesh(
            X,
            Y,
            Zbinned,
            cmap=discrete_cmap,
            norm=norm,
            shading="auto",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10, 100)
        ax.set_ylim(1e6, 1e7)
        ax.set_xlabel("Sherwood number (-)")
        ax.set_ylabel("Power number (-)")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())

        cbar = fig.colorbar(pc, ax=ax, ticks=levels, fraction=0.046, pad=0.04)
        cbar.set_label(label)
        cbar.ax.yaxis.set_major_formatter(ScalarFormatter())

        if cfd_prediction is not None:
            sh_cfd, pn_cfd = cfd_prediction
            ax.scatter(
                sh_cfd,
                pn_cfd,
                marker="*",
                color="white",
                edgecolor="black",
                s=150,
                label="Spacer performance based on CFD-derived correlations",
                zorder=10,
            )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.05),
            ncol=1,
            fontsize=14,
            frameon=False,
        )
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.9,
        bottom=0.25,
        wspace=0.3,
    )
    plt.suptitle(
        f"Performance Map: {''.join([c for c in os.path.basename(file_path) if c.isupper()])}",
        fontsize=20, y=0.98
    )
    plt.savefig(
        os.path.join(
            os.path.dirname(file_path),
            f"performance_map_{''.join([c for c in os.path.basename(file_path) if c.isupper()])}.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    file_path = os.path.join(
        os.path.dirname(__file__),
        "performance_map_BWRO_8x8_nfe2.xlsx",
    )
    cfd_prediction = (14.33722354, 3328175.7)  # Prediction from simulation
    zlims = {
        "LCOW": (0.16, 0.26, 10),
        "SEC": (0.7, 1.4, 10),
    }
    make_binned_heatmaps(file_path, cfd_prediction=cfd_prediction, zlims=zlims)

