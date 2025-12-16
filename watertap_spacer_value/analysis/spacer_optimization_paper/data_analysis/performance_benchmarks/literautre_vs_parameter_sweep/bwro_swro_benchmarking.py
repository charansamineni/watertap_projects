import os
from collections import OrderedDict
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import ScalarFormatter, NullFormatter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from watertap_spacer_value.analysis.spacer_optimization_paper.data_analysis.cost_savings_opportunities.overall_cost_savings_opportunities import set_plot_style, get_correlation_map
from watertap_spacer_value.plotting_tools.paul_tol_color_maps import get_tol_cmap
from watertap_spacer_value.analysis.spacer_optimization_paper.data_analysis.cost_savings_opportunities.overall_cost_savings_opportunities import get_correlation_map




def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def load_correlation_data(
    xlsx_path,
    *,
    avg_sheet="macro_results",
):
    """
    Load average points and trend lines from a correlation Excel file.

    Returns
    -------
    avg_df : pd.DataFrame
        Must contain ['correlation', 'Average Sh', 'Average Pn']
    trend_df : pd.DataFrame
        Columns: ['Sh', 'Pn', 'Sheet']
    """

    xls = pd.ExcelFile(xlsx_path)

    # -----------------------------
    # Load averages
    # -----------------------------
    if avg_sheet not in xls.sheet_names:
        raise KeyError(f"Missing `{avg_sheet}` sheet in {xlsx_path}")

    avg_df = pd.read_excel(xls, sheet_name=avg_sheet)

    if not {"correlation", "Average Sh", "Average Pn"}.issubset(avg_df.columns):
        raise KeyError(
            f"`{avg_sheet}` sheet must contain 'correlation', 'Average Sh', and 'Average Pn' columns."
        )

    # -----------------------------
    # Load trends
    # -----------------------------
    trend_frames = []

    for sheet in xls.sheet_names:
        if 'micro_trend' not in sheet.lower():
            continue
        df = pd.read_excel(xls, sheet_name=sheet)

        # Normalize columns
        df.columns = df.columns.str.strip()  # remove leading/trailing spaces
        if not {"Sh", "Pn"}.issubset(df.columns):
            continue
        name = sheet.replace("micro_trend", "").strip(" _-")
        trend_frames.append(
            df[["Sh", "Pn"]].assign(Sheet=name)
        )
    trend_df = (
        pd.concat(trend_frames, ignore_index=True)
        if trend_frames
        else pd.DataFrame(columns=["Sh", "Pn", "Sheet"])
    )
    return avg_df, trend_df


def plot_sh_pn_map(
    df,
    metric_col,
    *,
    metric_label=None,
    cmap="viridis",
    zlims=None,
    n_levels=10,
    ax=None,
):
    """
    Plot a binned Sherwood–Power number heatmap.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    metric_col : str
        Column name to plot (z-value).
    metric_label : str, optional
        Colorbar label.
    cmap : str or Colormap
        Matplotlib or Tol colormap.
    zlims : tuple (zmin, zmax), optional
        Color limits.
    n_levels : int
        Number of discrete color levels.
    ax : matplotlib axis, optional
        Axis to plot into.
    """

    if metric_col not in df.columns:
        raise KeyError(f"Column `{metric_col}` not found in DataFrame.")
    elif df[metric_col].isnull().all():
        raise ValueError(f"Column `{metric_col}` contains only NaN values.")
    elif "Sherwood number" not in df.columns or "Power number" not in df.columns:
        raise KeyError("DataFrame must contain 'Sherwood number' and 'Power number' columns.")

    x_all = df["Sherwood number"].astype(float).values
    y_all = df["Power number"].astype(float).values
    z_all = df[metric_col].astype(float).values

    mask = (
        np.isfinite(x_all)
        & np.isfinite(y_all)
        & np.isfinite(z_all)
        & (x_all > 0)
        & (y_all > 0)
    )

    if mask.sum() < 5:
        raise ValueError("Not enough valid data points after filtering.")

    x = x_all[mask]
    y = y_all[mask]
    z = z_all[mask]

    if "pressure drop" in metric_col.lower():
        z = -1e-5 * z  # Convert to bar and invert

    print(f"Max and min {metric_col}: {z.max()}, {z.min()}")

    # Unique sorted values
    sh_vals = np.unique(x)
    pn_vals = np.unique(y)

    if len(sh_vals) < 2 or len(pn_vals) < 2:
        raise ValueError("Need at least 2 unique Sh and Pn values.")

    # Log-spaced bin edges
    sh_edges = np.geomspace(sh_vals.min(), sh_vals.max(), len(sh_vals) + 1)
    pn_edges = np.geomspace(pn_vals.min(), pn_vals.max(), len(pn_vals) + 1)

    # Bin averaging
    H, _, _ = np.histogram2d(x, y, bins=[sh_edges, pn_edges], weights=z)
    counts, _, _ = np.histogram2d(x, y, bins=[sh_edges, pn_edges])

    Z = np.full_like(H, np.nan, dtype=float)
    valid = counts > 0
    Z[valid] = H[valid] / counts[valid]

    # Color limits
    if zlims is None:
        zmin = np.nanmin(Z[valid])
        zmax = np.nanmax(Z[valid])
    else:
        zmin, zmax = zlims

    levels = np.linspace(zmin, zmax, n_levels + 1)

    # Colormap handling
    if isinstance(cmap, str):
        base_cmap = plt.get_cmap(cmap)
    else:
        base_cmap = cmap

    discrete_cmap = ListedColormap(
        base_cmap(np.linspace(0, 1, n_levels))
    )
    norm = BoundaryNorm(levels, ncolors=n_levels)

    # Axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_box_aspect(1)

    X, Y = np.meshgrid(sh_edges, pn_edges, indexing="ij")

    pc = ax.pcolormesh(
        X, Y, Z,
        cmap=discrete_cmap,
        norm=norm,
        shading="auto",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sherwood number (-)", fontsize=12)
    ax.set_ylabel("Power number (-)", fontsize=12)

    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)

    cbar = plt.colorbar(pc, cax=cax, ticks=levels)
    # Determine formatted colorbar label
    if metric_label is not None:
        if "LCOW" in metric_label:
            cbar_label = r"LCOW ($/m^3$)"
        elif "SEC" in metric_label or "Specific energy consumption" in metric_label:
            cbar_label = r"SEC (kWh/m^3)"
        elif "Pressure drop" in metric_label:
            cbar_label = r"ΔP (bar)"
        elif "cp" in metric_label.lower():
            cbar_label = r"CP modulus (-)"
        else:
            cbar_label = metric_label
    else:
        cbar_label = metric_col

    cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=15, fontsize=12)
    return ax


def overlay_averages(
    ax,
    avg_df,
    correlation_map,
    *,
    sh_col="Average Sh",
    pn_col="Average Pn",
    label_col="correlation",
    size=80,
    zorder=10,
    add_legend=True,
):
    """
    Overlay average points with markers and colors from correlation_map.
    """

    handles = []
    labels = []

    for _, row in avg_df.iterrows():
        corr_key = row[label_col].lower()
        if corr_key in correlation_map:
            # color = correlation_map[corr_key]["color"]
            marker = correlation_map[corr_key]["marker"]
            label = correlation_map[corr_key]["label"]
        else:
            color = "white"
            marker = "o"
            label = row[label_col].capitalize()

        h = ax.scatter(
            row[sh_col],
            row[pn_col],
            marker=marker,
            s=size,
            facecolor="white",
            edgecolor="black",
            linewidth=1.2,
            zorder=zorder,
            label=label,
        )

        handles.append(h)
        labels.append(label)

    # Legend is handled separately in figure-level
    return ax


def overlay_trends(
    ax,
    trend_df,
    correlation_map,
    *,
    sh_col="Sh",
    pn_col="Pn",
    group_col="Sheet",
    linewidth=2,
    linestyles=None,
    zorder=9,
    add_legend=True,
):
    """
    Overlay Sherwood–Power trend lines on an axis using the correlation map.
    """

    if linestyles is None:
        linestyles = ["solid", "dashed", "dotted", "dashdot", (0, (3, 5, 1, 5))]

    handles = []
    labels = []

    for i, (name, grp) in enumerate(trend_df.groupby(group_col)):
        corr_key = name.lower()  # match correlation map keys
        if corr_key not in correlation_map:
            color = "black"
            ls = linestyles[i % len(linestyles)]
            label = name.title()
        else:
            # color = correlation_map[corr_key]["color"]
            color = "black"
            ls = linestyles[i % len(linestyles)]
            label = correlation_map[corr_key]["label"]

        h, = ax.plot(
            grp[sh_col],
            grp[pn_col],
            linestyle=ls,
            color=color,
            linewidth=linewidth,
            zorder=zorder,
            label=label,
        )
        handles.append(h)
        labels.append(label)

    if add_legend:
        unique = OrderedDict(zip(labels, handles))
        ax.legend(
            unique.values(),
            unique.keys(),
            loc="lower center",
            frameon=False,
            fontsize=8,
            ncol=2
        )

    return ax


def make_swro_bwro_summary_figure(
    *,
    swro_df,
    bwro_df,
    swro_avg_df,
    bwro_avg_df,
    swro_trend_df,
    zlims,
    cmap_map,
    correlation_map,
    figsize=(8.5, 12),
    marker_size=80,
):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.5, hspace=0.5)

    axs = np.empty((3, 2), dtype=object)
    for i in range(3):
        for j in range(2):
            axs[i, j] = fig.add_subplot(gs[i, j])
            axs[i, j].set_box_aspect(1)
            axs[i, j].set_xscale("log")
            axs[i, j].set_yscale("log")
            axs[i, j].set_xlim(10, 200)
            axs[i, j].set_ylim(1e5, 1e7)
            axs[i, j].tick_params(labelsize=12)

    # ------------------------------------------------------------------
    # (0,0) SWRO trends + averages
    # ------------------------------------------------------------------
    ax = axs[0, 0]
    overlay_trends(ax, swro_trend_df, correlation_map, add_legend=True)
    overlay_averages(ax, swro_avg_df, correlation_map, add_legend=False, size=marker_size)
    ax.set_xlabel("Sherwood number (-)", fontsize=12)
    ax.set_ylabel("Power number (-)", fontsize=12)
    ax.grid(False)

    # Add a fake color bar for consistency
    sm = plt.cm.ScalarMappable(cmap="gray", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.ax.set_ylabel("Correlation trends", rotation=270, labelpad=15)

    # ------------------------------------------------------------------
    # Remaining panels (maps)
    # ------------------------------------------------------------------
    panels = [
        ("Average pressure drop", "SWRO – Pressure drop", cmap_map["pressure_drop"], zlims["Pressure drop"], swro_avg_df, axs[0, 1]),
        ("Average cp (TDS)", "SWRO – CP modulus", cmap_map["cp"], zlims["Average cp (TDS)"], swro_avg_df, axs[1, 0]),
        ("Energy Consumption", "SWRO – Specific energy consumption", cmap_map["sec"], zlims["Energy Consumption"], swro_avg_df, axs[1, 1]),
        ("LCOW", "SWRO – LCOW", cmap_map["lcow"], (0.7, 1.0), swro_avg_df, axs[2, 0]),
        ("LCOW", "BWRO – LCOW", cmap_map["lcow"], zlims["LCOW"], bwro_avg_df, axs[2, 1]),
    ]

    for metric, title, cmap, zlim, avg_df, ax in panels:
        plot_sh_pn_map(swro_df if "SWRO" in title else bwro_df,
                       metric_col=metric,
                       metric_label=title.split("–")[-1].strip(),
                       cmap=cmap,
                       zlims=zlim,
                       ax=ax)
        overlay_averages(ax, avg_df, correlation_map, add_legend=False)

    # ------------------------------------------------------------------
    # Single legend for all correlations
    # ------------------------------------------------------------------
    fig_handles = []
    fig_labels = []
    for key, val in correlation_map.items():
        h = axs[0, 0].scatter([], [], marker=val["marker"], color="white", edgecolor="black", s=marker_size)
        fig_handles.append(h)
        fig_labels.append(val["label"])

    fig.legend(
        fig_handles,
        fig_labels,
        loc="upper center",
        ncol=len(fig_labels),
        fontsize=10,
        frameon=False,
    )

    return fig, axs


if __name__ == "__main__":
    correlation_map = get_correlation_map()
    set_plot_style()
    swro_data_path = f"../../../spacer_scale_benchmarks/sh_pn_sweeps/swro/swro_sweep_results.csv"
    bwro_sweep_path = f"../../../spacer_scale_benchmarks/sh_pn_sweeps/bwro/bwro_sweep_results.csv"
    swro_corr_path = "../../../spacer_scale_benchmarks/literature_correlations/SWRO_results.xlsx"
    bwro_corr_path = "../../../spacer_scale_benchmarks/literature_correlations/BWRO_results.xlsx"
    swro_avg_df, swro_trend_df = load_correlation_data(swro_corr_path)
    bwro_avg_df, bwro_trend_df = load_correlation_data(bwro_corr_path)

    swro_df = load_data(swro_data_path)
    bwro_df = load_data(bwro_sweep_path)

    fig, axs = make_swro_bwro_summary_figure(
        swro_df=swro_df,
        bwro_df=bwro_df,
        swro_avg_df=swro_avg_df,
        bwro_avg_df=bwro_avg_df,
        swro_trend_df=swro_trend_df,
        zlims={
            "Pressure drop": (0, 9),
            "Average cp (TDS)": (1, 2),
            "Energy Consumption": (3.4, 5.1),
            "LCOW": (0.24, 0.35),
        },
        cmap_map={
            "pressure_drop": get_tol_cmap("tol.red"),
            "cp": get_tol_cmap("tol.purple"),
            "sec": get_tol_cmap("tol.sunset-diverging"),
            "lcow": get_tol_cmap("tol.burd"),
        },
        correlation_map=correlation_map,
    )
    plt.savefig("swro_bwro_summary.svg", bbox_inches="tight")
    plt.show()



