import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from watertap_spacer_value.plotting_tools.paul_tol_color_maps import (
    gen_paultol_colormap,
)

# -----------------------------------------------------------------------------
# Plot settings and utilities
# -----------------------------------------------------------------------------

STRATEGY_ORDER = [
    "spacer_module_system",
    "spacer_system",
    "module_system",
    "spacer_module",
    "system",
    "module",
    "spacer",
]


def format_strategy_label(strategy):
    parts = [p.capitalize() for p in strategy.split("_")]
    return parts[0] if len(parts) == 1 else "\n+".join(parts)


def set_plot_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": "Arial",
            "mathtext.default": "regular",
            "svg.fonttype": "none",
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "axes.linewidth": 0.8,
            "xtick.major.size": 4.0,
            "xtick.minor.size": 2.0,
            "ytick.major.size": 4.0,
            "ytick.minor.size": 2.0,
            "xtick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.major.width": 0.8,
            "ytick.minor.width": 0.6,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "patch.linewidth": 1.0,
        }
    )


def get_correlation_map():
    colors = gen_paultol_colormap("tol_bright", 5)
    return {
        "dacosta": {"color": colors[0], "label": "DaCosta et al.", "marker": "o"},
        "guillen": {"color": colors[1], "label": "Guillen et al.", "marker": "s"},
        "koustou": {"color": colors[2], "label": "Koustou et al.", "marker": "D"},
        "kuroda": {"color": colors[3], "label": "Kuroda et al.", "marker": "^"},
        "schock": {"color": colors[4], "label": "Schock et al.", "marker": "v"},
    }


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------

def prepare_plotting_data():
    for case in ["bwro", "swro"]:
        results = []

        for correlation in ["guillen", "schock", "dacosta", "koustou", "kuroda"]:
            file = (
                f"../../multi_scale_optimization/{case}/"
                f"ro_optimization_strategies_results_{case}_{correlation}.xlsx"
            )
            df = pd.read_excel(file, sheet_name="macro_trends")

            base_lcow = df.loc[df["Unnamed: 0"] == "simulation", "LCOW"].iloc[0]
            df["pct_savings"] = 100 * (base_lcow - df["LCOW"]) / base_lcow

            df = df[["Unnamed: 0", "LCOW", "pct_savings"]]
            df = df.rename(columns={"Unnamed: 0": "strategy"})
            df["correlation"] = correlation

            results.append(df)

        pd.concat(results, ignore_index=True).to_csv(
            f"{case}_cost_savings_opportunities.csv", index=False
        )


def load_plotting_dataframe():
    swro = pd.read_csv("swro_cost_savings_opportunities.csv")
    bwro = pd.read_csv("bwro_cost_savings_opportunities.csv")

    swro["process"] = "swro"
    bwro["process"] = "bwro"

    df = pd.concat([swro, bwro], ignore_index=True)
    df = df[df["strategy"] != "simulation"].copy()

    df["strategy_label"] = df["strategy"].apply(format_strategy_label)
    df["process"] = df["process"].str.lower()
    df["correlation"] = df["correlation"].str.lower()

    return df


# -----------------------------------------------------------------------------
# Bar plot
# -----------------------------------------------------------------------------

def plot_cost_savings_barplot(df):
    set_plot_style()
    corr_map = get_correlation_map()
    order_labels = [format_strategy_label(s) for s in STRATEGY_ORDER]

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 1, hspace=0.35)
    axes = [fig.add_subplot(gs[i]) for i in range(2)]

    for ax, process in zip(axes, ["swro", "bwro"]):
        data = df[df["process"] == process]

        sns.barplot(
            data=data,
            x="strategy_label",
            y="pct_savings",
            hue="correlation",
            order=order_labels,
            palette={k: v["color"] for k, v in corr_map.items()},
            edgecolor="black",
            linewidth=1,
            ax=ax,
        )

        ax.set_title(process.upper(), fontsize=14)
        ax.set_ylabel("Cost savings opportunity (%)", fontsize=12)
        ax.set_xlabel("")
        ax.set_ylim(0, 30)
        ax.legend_.remove()

    legend_handles = [
        Patch(
            facecolor=v["color"],
            edgecolor="black",
            label=v["label"],
        )
        for v in corr_map.values()
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.subplots_adjust(bottom=0.15)
    plt.savefig("swro_bwro_pct_savings_barplot.svg", dpi=300)
    plt.show()


# -----------------------------------------------------------------------------
# Box plot (min–max + raw points)
# -----------------------------------------------------------------------------
def plot_cost_savings_boxplot(df):
    processes = ["swro", "bwro"]
    y_label = "Cost savings opportunity (%)"
    y_lim = (0, 30)
    jitter_width = 0.15
    rng = np.random.default_rng(42)  # reproducible jitter

    order_labels = [format_strategy_label(s) for s in STRATEGY_ORDER]

    corr_map = get_correlation_map()
    ordered_corrs = list(corr_map.keys())

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 1, hspace=0.35)
    axes = [fig.add_subplot(gs[i]) for i in range(2)]

    for ax, process in zip(axes, processes):
        data_proc = df[df["process"] == process]
        print(f"Processing {process}:")
        print("Order labels:", order_labels)
        print("Unique strategy_label in df:", data_proc["strategy_label"].unique())

        data_matrix = []
        for label in order_labels:
            row = []
            for corr in ordered_corrs:
                subset = data_proc[
                    (data_proc["strategy_label"] == label) & (data_proc["correlation"] == corr)
                ]["pct_savings"].values
                if len(subset) == 0:
                    print(f"Missing data for strategy={label}, correlation={corr}")
                    row.append(np.nan)
                else:
                    row.append(subset[0])
            data_matrix.append(row)
        data_matrix = np.array(data_matrix)
        x_positions = np.arange(1, len(order_labels) + 1)
        valid_cols = ~np.isnan(data_matrix).all(axis=0)
        data_matrix = data_matrix[:, valid_cols]

        ax.boxplot(
            data_matrix.T,
            positions=x_positions,
            patch_artist=True,
            showfliers=False,
            whis=(0, 100),
            boxprops=dict(facecolor="none", edgecolor="black", linewidth=1.3),
            whiskerprops=dict(color="black", linewidth=1.3),
            capprops=dict(color="black", linewidth=1.3),
            medianprops=dict(color="black", linewidth=1.5),
        )

        for i, corr in enumerate(np.array(ordered_corrs)[valid_cols]):
            y_vals = data_matrix[:, i]
            x_jitter = x_positions + rng.uniform(
                -jitter_width, jitter_width, size=len(y_vals)
            )

            ax.scatter(
                x_jitter,
                y_vals,
                marker=corr_map[corr]["marker"],
                color=corr_map[corr]["color"],
                edgecolor="black",
                linewidth=0.6,
                s=70,
                alpha=0.9,
                zorder=3,
            )

        ax.set_title(process.upper(), fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_ylim(*y_lim)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(order_labels, fontsize=12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    legend_handles = [
        Line2D(
            [0], [0],
            marker=v["marker"],
            color=v["color"],
            linestyle="None",
            markersize=7,
            markeredgecolor="black",
            label=v["label"],
        )
        for v in corr_map.values()
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.subplots_adjust(bottom=0.15)
    fig.savefig("swro_bwro_pct_savings_boxplot.svg", dpi=300)
    plt.show()


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    prepare_plotting_data()
    df = load_plotting_dataframe()
    plot_cost_savings_barplot(df)
    plot_cost_savings_boxplot(df)
