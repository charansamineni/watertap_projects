import os
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import ScalarFormatter, NullFormatter
import matplotlib.colors as mcolors


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np

import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def get_tol_cmap(name, *, discrete=None, reverse=False, as_hex=False):
    """
    Return a Paul Tol colormap (qualitative, sequential, diverging, rainbow, cyclic).

    Parameters
    ----------
    name : str
        Name of Tol color scheme. Accepts an optional `tol.` prefix.
    discrete : int, optional
        If given, returns a ListedColormap with N discrete colors.
    reverse : bool, optional
        Reverse color order.
    as_hex : bool, optional
        If True, return a list of hex colors instead of a cmap.
    """

    # -------------------------------------------------------
    # Normalize name
    # -------------------------------------------------------
    if isinstance(name, str) and name.startswith("tol."):
        name = name[4:]
    name = str(name).lower()

    # -------------------------------------------------------
    # Official Paul Tol color definitions (SRON 2021)
    # -------------------------------------------------------

    qualitative = {
        "bright": [
            "#4477AA", "#66CCEE", "#228833", "#CCBB44",
            "#EE6677", "#AA3377", "#BBBBBB"
        ],
        "vibrant": [
            "#0077BB", "#33BBEE", "#009988", "#EE7733",
            "#CC3311", "#EE3377", "#BBBBBB"
        ],
        "muted": [
            "#332288", "#88CCEE", "#44AA99", "#117733",
            "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"
        ],
        "pale": [
            "#BBCCEE", "#CCEEFF", "#CCDDAA", "#EEEEBB",
            "#FFCCCC", "#DDDDDD"
        ],
        "dark": [
            "#222255", "#225555", "#225522",
            "#666633", "#663333", "#552222"
        ],
        "light": [
            "#77AADD", "#99DDFF", "#44BB99",
            "#BBCC33", "#EE8866", "#EEDD88",
            "#FFAABB", "#AAAAAA"
        ],
    }

    # -------------------------------------------------------
    # Sequential color schemes (Tol 2021)
    # -------------------------------------------------------

    sequential = {
        "blue": [
            "#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1",
            "#6BAED6", "#4292C6", "#2171B5", "#084594"
        ],
        "green": [
            "#F7FCF5", "#E5F5E0", "#C7E9C0", "#A1D99B",
            "#74C476", "#41AB5D", "#238B45", "#005A32"
        ],
        "red": [
            "#FFF5F0", "#FEE0D2", "#FCBBA1", "#FC9272",
            "#FB6A4A", "#EF3B2C", "#CB181D", "#99000D"
        ],
        "purple": [
            "#FCFBFD", "#EFEDF5", "#DADAEB", "#BCBDDC",
            "#9E9AC8", "#807DBA", "#6A51A3", "#4A1486"
        ],
        "orange": [
            "#FFF5EB", "#FEE6CE", "#FDD0A2", "#FDAE6B",
            "#FD8D3C", "#F16913", "#D94801", "#8C2D04"
        ],
        # Tol "Smooth Rainbow"
        "smoothrainbow": [
            "#E8ECFB", "#D9CCE3", "#D1BBD7", "#CAACCB", "#BA8DB4",
            "#AE76A3", "#AA559F", "#A24D99", "#9F4797", "#994091",
            "#8E3B91", "#7C358F", "#6D2C91", "#5E2592", "#4E1D92"
        ],
        # Tol "Discrete Rainbow"
        "rainbow_discrete": [
            "#781C81", "#3F37A2", "#3465A4", "#1A92C7",
            "#11B1D8", "#0ECAD8", "#35D5C4", "#66DEA3",
            "#98E482", "#CAE65D", "#F9E349"
        ],
        # Tol "Sunset" (sequential)
        "sunset": [
            "#364B9A", "#4A7BB7", "#6EA6CD", "#98CAE1",
            "#C2E4EF", "#EAECCC", "#FEDA8B", "#FDB366",
            "#F67E4B", "#DD3D2D", "#A50026"
        ],
    }

    # -------------------------------------------------------
    # Diverging color schemes
    # -------------------------------------------------------
    diverging = {
        "sunset-diverging": [
            "#364B9A", "#4A7BB7", "#6EA6CD", "#98CAE1",
            "#C2E4EF", "#EAECCC", "#FEDA8B", "#FDB366",
            "#F67E4B", "#DD3D2D", "#A50026"
        ],
        "burd": [
            "#2166AC", "#4393C3", "#92C5DE", "#D1E5F0",
            "#F7F7F7", "#FDDBC7", "#F4A582", "#D6604D",
            "#B2182B"
        ],
        "prgn": [
            "#762A83", "#9970AB", "#C2A5CF",
            "#E7D4E8", "#F7F7F7", "#D9F0D3", "#ACD39E",
            "#5AAE61", "#1B7837", "#00441B"
        ],
    }

    # -------------------------------------------------------
    # Cyclic
    # -------------------------------------------------------
    cyclic = {
        "cyclic": [
            "#5E4FA2", "#3288BD", "#66C2A5", "#ABDDA4",
            "#E6F598", "#FEE08B", "#FDAE61", "#F46D43",
            "#D53E4F", "#9E0142", "#5E4FA2"
        ]
    }

    # -------------------------------------------------------
    # Combine all
    # -------------------------------------------------------
    all_maps = {
        **qualitative,
        **sequential,
        **diverging,
        **cyclic
    }

    # Aliases for convenience
    aliases = {
        "rainbow": "smoothrainbow",
        "rainbow_smooth": "smoothrainbow",
        "rainbow_dis": "rainbow_discrete",
        "diverging": "sunset-diverging",
        "seq": "blue",
    }

    if name in aliases:
        name = aliases[name]

    # -------------------------------------------------------
    # Validate and fetch
    # -------------------------------------------------------
    if name not in all_maps:
        raise ValueError(
            f"Unknown Tol colormap `{name}`. Valid names:\n{sorted(all_maps.keys())}"
        )

    colors = all_maps[name]

    # Reverse?
    if reverse:
        colors = list(reversed(colors))

    # -------------------------------------------------------
    # Hex output only
    # -------------------------------------------------------
    if as_hex:
        if discrete is not None:
            idx = np.linspace(0, len(colors) - 1, discrete).astype(int)
            return [colors[i] for i in idx]
        return list(colors)

    # -------------------------------------------------------
    # Discrete colormap
    # -------------------------------------------------------
    if discrete is not None:
        idx = np.linspace(0, len(colors) - 1, discrete).astype(int)
        sel = [colors[i] for i in idx]
        return ListedColormap(sel, name=f"{name}_{discrete}")

    # -------------------------------------------------------
    # Continuous colormap for sequential/diverging/cyclic
    # -------------------------------------------------------
    if name in sequential or name in diverging or name in cyclic:
        return LinearSegmentedColormap.from_list(name, colors)

    # Qualitative always discrete
    return ListedColormap(colors, name=name)



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
    trend_df=None,
    overlay=None,
):
    df = load_data(file_path)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": "Arial",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "mathtext.default": "regular",
        "svg.fonttype": "none",
    })

    # -------------------------------------------------------------------
    # DEFAULT LIMITS
    # -------------------------------------------------------------------
    if zlims is None:
        zlims = {
            "LCOW": (0.5, 3, 10),
            "Energy Consumption": (2, 7, 10),
            "Pressure drop": (0.1, 2, 10),
        }

    # -------------------------------------------------------------------
    # DEFAULT METRIC GROUPS
    # -------------------------------------------------------------------
    # Updated with Paul Tol colormaps
    if metric_groups is None:
        metric_groups = [
            [
                ("Energy Consumption", "Specific energy consumption (kWh/m³)", "tol.sunset-diverging"),
                ("LCOW", "Levelized cost of water ($/m³)", "tol.burd"),
            ],
            [
                ("Operating pressure", "Operating pressure (bar)", "tol.orange"),
                ("Average telescoping potential", "$\Delta P$ per length (bar/m)", "tol.red"),
                ("Average cp (TDS)", "Average cp modulus (-)", "tol.purple"),
            ],
        ]

    # -------------------------------------------------------------------
    # Filter data for correct porosity + channel height
    # -------------------------------------------------------------------
    subset = df[
        (df["Spacer porosity"] == por) &
        (df["Channel height"] == ch)
    ].copy()

    if subset.empty:
        raise ValueError("No data available for specified porosity & channel height.")

    x = subset["Sherwood number"].values
    y = subset["Power number"].values

    sh_vals = np.sort(np.unique(x))
    pn_vals = np.sort(np.unique(y))

    if len(sh_vals) < 2 or len(pn_vals) < 2:
        raise ValueError("Not enough unique Sh/Pn values for binning.")

    # Log-spaced bin edges
    sh_edges = np.geomspace(sh_vals[0], sh_vals[-1], len(sh_vals) + 1)
    pn_edges = np.geomspace(pn_vals[0], pn_vals[-1], len(pn_vals) + 1)

    # -------------------------------------------------------------------
    # BEGIN METRIC LOOPS
    # -------------------------------------------------------------------
    for group_index, metrics in enumerate(metric_groups):
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + 2) // 3

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 6 * n_rows),
            sharex=False, sharey=False
        )
        axes = axes.flatten()

        for ax in axes:
            ax.set_box_aspect(1)
            ax.tick_params(axis='y', labelsize=12)
            ax.tick_params(axis='x', labelsize=12)

        for ax, (metric, label, cmap_name) in zip(axes, metrics):

            z = subset[metric].copy().values

            # Unit conversions
            if metric == "Pressure drop":
                z *= -1e-5
            elif metric == "Operating pressure":
                z *= 1e-5

            # Print raw z min/max for manual range selection
            try:
                z_min = float(np.nanmin(z))
                z_max = float(np.nanmax(z))
            except ValueError:
                z_min, z_max = None, None
            print(f"Metric: {metric} | raw z min: {z_min:.2f} | raw z max: {z_max:.2f}")

            # -------------------------------------------------------------------
            # BINNING PROCESS
            # -------------------------------------------------------------------
            H, _, _ = np.histogram2d(x, y, bins=[sh_edges, pn_edges], weights=z)
            counts, _, _ = np.histogram2d(x, y, bins=[sh_edges, pn_edges])

            Zbinned = np.full_like(H, np.nan)
            valid = counts > 0
            Zbinned[valid] = H[valid] / counts[valid]

            # Print binned z min/max (ignoring empty bins)
            if np.any(valid):
                bmin = float(np.nanmin(Zbinned[valid]))
                bmax = float(np.nanmax(Zbinned[valid]))
            else:
                bmin, bmax = None, None
            print(f"Metric: {metric} | binned z min: {bmin:.2f} | binned z max: {bmax:.2f}")

            # Mesh for pcolormesh
            X, Y = np.meshgrid(sh_edges, pn_edges, indexing="ij")

            # -------------------------------------------------------------------
            # COLOR MAP HANDLING (continuous → discrete)
            # -------------------------------------------------------------------
            if cmap_name.startswith("tol."):
                base_cmap = get_tol_cmap(cmap_name)
            else:
                base_cmap = plt.get_cmap(cmap_name)

            zmin, zmax, n_levels = zlims.get(metric)
            levels = np.linspace(zmin, zmax, n_levels + 1)

            discrete_cmap = ListedColormap(
                base_cmap(np.linspace(0, 1, n_levels))
            )
            norm = BoundaryNorm(levels, ncolors=n_levels)

            # -------------------------------------------------------------------
            # DRAW PLOT
            # -------------------------------------------------------------------
            pc = ax.pcolormesh(
                X, Y, Zbinned,
                cmap=discrete_cmap,
                norm=norm,
                shading="auto",
            )

            # Axis formatting
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(10, 200)
            ax.set_ylim(1e5, 1e7)
            #
            # Disable autoscaling so limits remain manual
            ax.set_autoscalex_on(False)
            ax.set_autoscaley_on(False)

            ax.set_xlabel("Sherwood number (-)")
            ax.set_ylabel("Power number (-)")

            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())

            # Colorbar
            cbar = fig.colorbar(pc, ax=ax, ticks=levels, fraction=0.046, pad=0.04)
            cbar.set_label(label, rotation=270, labelpad=15)
            cbar.ax.yaxis.set_major_formatter(ScalarFormatter())

            # -------------------------------------------------------------------
            # APPLY OVERLAYS (unchanged)
            # -------------------------------------------------------------------
            if overlay == "average" and correlation_df is not None:
                marker_styles = ["o", "s", "D", "^", "v"]
                scatter_handles = []
                scatter_labels = []
                for i, (_, row) in enumerate(correlation_df.iterrows()):
                    marker = marker_styles[i % len(marker_styles)]
                    lbl = f"{row['correlation'].capitalize()} et al."
                    h = ax.scatter(
                        row["Average Sh"], row["Average Pn"],
                        color="white", edgecolor="black",
                        marker=marker, s=120, zorder=10, label=lbl
                    )
                    scatter_handles.append(h)
                    scatter_labels.append(lbl)

                unique = OrderedDict(
                    sorted(
                        zip(scatter_labels, scatter_handles),
                        key=lambda x: x[0]
                    )
                )
                fig.legend(
                    unique.values(),
                    unique.keys(),
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0),
                    ncol=len(unique),
                    fontsize=12,
                    frameon=False,
                )

            elif overlay == "trend" and trend_df is not None:
                line_styles = ["solid", "dotted", "dashed", "dashdot", (0, (3, 5, 1, 5))]
                for i, (sheet, group) in enumerate(trend_df.groupby("Sheet")):
                    ls = line_styles[i % len(line_styles)]
                    label_text = sheet.title()
                    ax.plot(
                        group["Sh"], group["Pn"],
                        linestyle=ls, color="black", linewidth=2,
                        label=label_text, zorder=10,
                    )
                    ax.legend(
                        loc="best",
                        fontsize=10,
                        frameon=False,
                    )
                # Legend for trends
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    unique = OrderedDict(
                        sorted(
                            zip(labels, handles),
                            key=lambda x: x[0]
                        )
                    )
                    fig.legend(
                        unique.values(),
                        unique.keys(),
                        loc="lower center",
                        bbox_to_anchor=(0.5, 0),
                        ncol=len(unique),
                        fontsize=12,
                        frameon=False,
                    )

        # Layout + save
        plt.subplots_adjust(
            left=0.05, right=0.95, top=0.95,
            bottom=0.25, wspace=0.3
        )
        plt.savefig(
            f"../sh_pn_sweeps/swro/heatmap_{overlay}_{group_index+1}.svg", format="svg",
            dpi=300, bbox_inches="tight", transparent=True
        )
        plt.show()




if __name__ == "__main__":
    file_path = "swro/swro_sweep_results.csv"

    xls = pd.ExcelFile("../literature_correlations/SWRO_results.xlsx")
    correlation_df = pd.read_excel(xls, sheet_name=0)

    trend_frames = []
    for sheet in xls.sheet_names:
        if "micro_trend" in sheet:
            df_tmp = pd.read_excel(xls, sheet_name=sheet)
            if {"Sh", "Pn"}.issubset(df_tmp.columns):
                name = sheet.replace("micro_trend", "").strip(" _-")
                trend_frames.append(df_tmp[["Sh", "Pn"]].assign(Sheet=name))

    trend_df = pd.concat(trend_frames, ignore_index=True) if trend_frames else \
        pd.DataFrame(columns=["Sh", "Pn"])

    make_binned_heatmaps(
        file_path=file_path,
        por=0.85,
        ch=0.7112e-3,
        zlims={
            "Operating pressure": (
                70,
                110,
                10,
            ),
            "Energy Consumption": (3, 5, 10),
            "LCOW": (0.7, 1, 10),
            "Average telescoping potential": (0, 1.1, 10),
            "Average cp (TDS)": (1, 1.7, 10),
        },
        metric_groups=None,
        trend_df=trend_df,
        overlay="average",
        correlation_df=correlation_df,
    )


