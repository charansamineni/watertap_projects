import os
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import ScalarFormatter, NullFormatter



def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def make_binned_heatmaps(
    file_path,
    por=0.85,
    ch=0.7112e-3,
    zlims=None,
    metric_groups=None,
    correlation_df=None,
):
    df = load_data(file_path)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 16

    if zlims is None:
        zlims = {
            "LCOW": (0.5, 3),
            "Energy Consumption": (2, 7),
            "Pressure drop": (0.1, 2),
        }

    if metric_groups is None:
        metric_groups = [
            [
                (
                    "Energy Consumption",
                    "Specific energy consumption (kWh/m³)",
                    "viridis",
                ),
                ("LCOW", "Levelized cost of water ($/m³)", "RdYlBu_r"),
            ],
            [
                ("Operating pressure", "Operating pressure (bar)", "cividis"),
                (
                    "Average telescoping potential",
                    "$\Delta P$ per length (bar/m)",
                    "PRGn_r",
                ),
                ("Average cp (TDS)", "Average cp modulus (-)", "Spectral_r"),
                # (
                #     "Outlet interface concentration",
                #     "Outlet interface concentration (g/L)",
                #     "viridis",
                # ),
            ],
        ]

    for group_idx, metrics in enumerate(metric_groups):
        n_metrics = len(metrics)
        if n_metrics < 3:
            n_cols = n_metrics
            n_rows = 1
        else:
            n_cols = 3
            n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), sharex=False, sharey=False
        )
        axes = axes.flatten() if n_metrics > 1 else [axes]

        subset = df[
            (df["Spacer porosity"] == por) & (df["Channel height"] == ch)
        ].copy()
        if subset.empty:
            raise ValueError(
                "No data available for the specified porosity and channel height."
            )

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

            if metric == "Pressure drop":
                z *= -1e-5
            elif metric == "Operating pressure":
                z *= 1e-5

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
            ax.set_xlim(10, 200)
            ax.set_ylim(1e5, 1e7)
            ax.set_xlabel("Sherwood number (-)")
            ax.set_ylabel("Power number (-)")
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())


            cbar = fig.colorbar(pc, ax=ax, ticks=levels, fraction=0.046, pad=0.04)
            cbar.set_label(label)
            cbar.ax.yaxis.set_major_formatter(ScalarFormatter())

            # Overlay literature correlations
            marker_styles = ["o", "s", "D", "^", "v"]
            scatter_handles = []
            scatter_labels = []

            for i, (_, row) in enumerate(correlation_df.iterrows()):
                marker_style = marker_styles[i % len(marker_styles)]
                translated_label = f"{row['correlation'].capitalize()} et al."
                scatter_handle = ax.scatter(
                    row["Average Sh"],
                    row["Average Pn"],
                    marker=marker_style,
                    color="white",
                    edgecolor="black",
                    s=80,
                    label=translated_label,
                    zorder=10,
                )
                scatter_handles.append(scatter_handle)
                scatter_labels.append(translated_label)

            # Build legend
            unique_labels_handles = OrderedDict(
                sorted(
                    ((label, handle) for label, handle in zip(scatter_labels, scatter_handles)),
                    key=lambda x: x[0]
                )
            )

            fig.legend(
                unique_labels_handles.values(),
                unique_labels_handles.keys(),
                loc="lower center",
                bbox_to_anchor=(0.5, 0),
                ncol=len(unique_labels_handles),
                fontsize=16,
                frameon=False,
            )

        plt.subplots_adjust(
            left=0.05,
            right=0.95,
            top=0.95,
            bottom=0.25,
            wspace=0.3,
        )
        plt.savefig(
            f"../sh_pn_sweeps/bwro/heatmap{group_idx + 1}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()


if __name__ == "__main__":
    file_path = "bwro/bwro_sweep_results.csv"
    correlation_df = pd.read_excel("../correlation_comparison/BWRO_results.xlsx")
    print(correlation_df.head())
    # Define label translations
    make_binned_heatmaps(
        file_path=file_path,
        por=0.85,
        ch=0.8636e-3,
        zlims={
            "Operating pressure": (
                24,
                42,
                10,
            ),
            "Energy Consumption": (0.85, 1.65, 10),
            "LCOW": (0.16, 0.27, 10),
            "Average telescoping potential": (0, 1.1, 10),
            "Average cp (TDS)": (1, 2, 10),
        },
        metric_groups=None,
        correlation_df=correlation_df,
    )

