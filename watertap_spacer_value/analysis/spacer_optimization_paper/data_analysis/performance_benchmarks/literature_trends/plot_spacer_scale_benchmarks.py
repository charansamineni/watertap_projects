import os
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import gridspec
import matplotlib.ticker as mticker
import pandas as pd
from watertap_spacer_value.analysis.spacer_optimization_paper.data_analysis.cost_savings_opportunities.overall_cost_savings_opportunities import set_plot_style, get_correlation_map


def plot_overall_figure(df_path="SWRO_results.xlsx", y_lim="auto", y_lims_manual=None):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": "Arial",
        "mathtext.default": "regular",
        "svg.fonttype": "none",
    })

    set_plot_style()

    # Read all sheets from the Excel file
    df_dict = pd.read_excel(df_path, sheet_name=None)

    # Create a figure with subplots
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.4, hspace=0.4)
    axes = [fig.add_subplot(gs[i]) for i in range(6)]

    # Set aspect ratio as square for all subplots
    for ax in axes:
        ax.set_box_aspect(1)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12)

    # Plot configuration
    micro_plot_keys = ["K", "cp_modulus", "f", "deltaP"]
    ylabels = [
        "Mass transfer coefficient (m/s)",
        "CP modulus (-)",
        "Friction factor (-)",
        "Pressure drop (bar)"
    ]

    correlation_map = get_correlation_map()

    for idx, key in enumerate(micro_plot_keys):
        row, col = divmod(idx, 2)
        ax = axes[row * 3 + col]

        for sheet_name, df in df_dict.items():
            if "micro_trend" in sheet_name:
                corr_key = sheet_name.split("_")[0]
                if corr_key in correlation_map:
                    if key == "deltaP":
                        dl = np.diff([0] + df["global_length_domain"].tolist())
                        dp_dx = df[key].tolist()
                        dp_dx = [dp / 1e5 for dp in dp_dx]  # Convert Pa to bar
                        dp = [dx * dpdx for dx, dpdx in zip(dl, dp_dx)]
                        total_dP = -1 * np.cumsum([0] + dp)
                        df[key] = total_dP[1:]

                    ax.scatter(
                        df["global_length_domain"],
                        df[key],
                        color=correlation_map[corr_key]["color"],
                        label=correlation_map[corr_key]["label"],
                        marker=correlation_map[corr_key]["marker"],
                        s=15
                    )
                    length_of_train = int(np.ceil(df["global_length_domain"].values[-1]))


        # X-axis configuration
        ax.set_xlim(0, length_of_train)
        if length_of_train == 6:
            ax.set_xticks(np.linspace(0, length_of_train, length_of_train + 1))
        elif length_of_train == 18:
            ax.set_xticks(np.linspace(0, length_of_train, length_of_train // 2 + 1))
            ax.axvline(length_of_train / 3, color="black", linestyle="--", linewidth=1)
            ax.axvline(2 * length_of_train / 3, color="black", linestyle="--", linewidth=1)
            # Stage annotations
            for stage, pos in zip(["Stage 1", "Stage 2", "Stage 3"],
                                  [length_of_train / 6, length_of_train / 2, 5 * length_of_train / 6]):
                ax.annotate(stage, xy=(pos, 0.9), xycoords=("data", "axes fraction"),
                            ha="center", va="bottom", fontsize=12, fontname="Arial")

        ax.set_xlabel("Axial position along RO system (m)", fontsize=12, fontname="Arial")
        ax.set_ylabel(ylabels[idx], fontsize=12, fontname="Arial")

        # Y-axis configuration
        if y_lim == "auto":
            all_vals = []
            for sheet_name, df in df_dict.items():
                if "micro_trend" in sheet_name:
                    all_vals.append(df[key].values)
            all_vals = np.concatenate(all_vals)
            yl_min, yl_max = 0, np.max(all_vals)
            scale = 10 ** np.floor(np.log10(yl_max - all_vals.min()))
            yl_max_rounded = np.ceil(yl_max / scale) * scale
            ax.set_ylim(yl_min, yl_max_rounded)
            ax.set_yticks(np.linspace(yl_min, yl_max_rounded, 6))
        elif y_lim == "manual":
            ax.set_ylim(*y_lims_manual[idx])
            ax.set_yticks(np.linspace(*y_lims_manual[idx], 6))

        if key in ["K"]:
            formatter = mticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(formatter)

    # Bar plots (SEC and LCOW)
    sec_ax, lcow_ax = axes[2], axes[5]
    macro_df = pd.read_excel(df_path, sheet_name="macro_results")
    correlations = [c for c in correlation_map if c in macro_df["correlation"].unique()]

    # SEC Bar plot
    sec_vals = [macro_df[macro_df["correlation"] == c]["Energy Consumption"].values[0] for c in correlations]
    sec_ax.bar(correlations,
               sec_vals,
               edgecolor="black",
               linewidth=1,
               color=[correlation_map[c]["color"] for c in correlations])
    sec_ax.set_xticks(range(len(correlations)))
    sec_ax.set_xticklabels(
        [correlation_map[c]["label"].replace(" et al.", "") for c in correlations],
        fontdict={'fontsize': 12, 'fontname': 'Arial'}, rotation=30, ha='right'
    )
    sec_ax.set_ylabel("SEC (kWh/m³)", fontsize=12, fontname="Arial")

    if y_lim == "manual":
        sec_ax.set_ylim(*y_lims_manual[4])
        sec_ax.set_yticks(np.linspace(*y_lims_manual[4], 6))
    elif y_lim == "auto":
        yl_max = np.max(sec_vals)
        scale = 10 ** (np.floor(np.log10(yl_max)))
        yl_max_rounded = np.ceil(yl_max / scale * 10) * scale / 10
        sec_ax.set_ylim(0, yl_max_rounded)
        sec_ax.set_yticks(np.linspace(0, yl_max_rounded, 6))

    # LCOW Bar plot
    lcow_vals = [macro_df[macro_df["correlation"] == c]["LCOW"].values[0] for c in correlations]
    lcow_ax.bar(correlations,
                lcow_vals,
                edgecolor="black",
                color=[correlation_map[c]["color"] for c in correlations])
    lcow_ax.set_xticks(range(len(correlations)))
    lcow_ax.set_xticklabels(
        [correlation_map[c]["label"].replace(" et al.", "") for c in correlations],
        fontdict={'fontsize': 12, 'fontname': 'Arial'}, rotation=30, ha='right'
    )
    lcow_ax.set_ylabel("LCOW ($/m³)", fontsize=12, fontname="Arial")
    if y_lim == "manual":
        lcow_ax.set_ylim(*y_lims_manual[5])
        lcow_ax.set_yticks(np.linspace(*y_lims_manual[5], 6))
    elif y_lim == "auto":
        yl_max = np.max(lcow_vals)
        scale = 10 ** np.floor(np.log10(yl_max))
        yl_max_rounded = np.ceil(yl_max / scale) * scale
        lcow_ax.set_ylim(0, yl_max_rounded)
        lcow_ax.set_yticks(np.linspace(0, yl_max_rounded, 6))

    # Add legend
    fig.legend(
        handles=[
            plt.Line2D([0], [0],
                       color=correlation_map[c]["color"],
                       lw=2,
                       label=correlation_map[c]["label"],
                       marker=correlation_map[c]["marker"])
            for c in correlation_map
        ],
        loc='lower center',
        bbox_to_anchor=(0.5, 0.04),
        ncol=len(correlation_map),
        frameon=False,
        handlelength=2.5,
        prop={'family': 'Arial', 'size': 12}
    )

    plt.subplots_adjust(bottom=0.15)

    # Extract "BWRO" or "SWRO"
    if "BWRO" in df_path:
        save_path = os.getcwd() + "/BWRO_correlation_trends.svg"
    elif "SWRO" in df_path:
        save_path = os.getcwd() + "/SWRO_correlation_trends.svg"
    else:
        raise ValueError("df_path must contain either 'BWRO' or 'SWRO' to determine save path.")
    plt.savefig(save_path, format="svg", dpi=300, transparent=True)
    plt.show()


if __name__ == "__main__":
    y_lims_swro = [
        (0, 2.5e-4),      # For K
        (1, 1.75),         # For cp_modulus
        (0, 5),   # For f
        (0, 3),          # For deltaP
        (0, 4.8),        # For SEC
        (0, 1.2)         # For LCOW
    ]
    y_lims_bwro = [
        (0, 2.5e-4),      # For K
        (1, 1.75),         # For cp_modulus
        (0, 5),   # For f
        (0, 3),         # For deltaP
        (0, 1.2),        # For SEC
        (0, 0.3)         # For LCOW
    ]

    plot_overall_figure(df_path=f"../../../spacer_scale_benchmarks/literature_correlations/SWRO_results.xlsx", y_lim="manual", y_lims_manual=y_lims_swro)
    plot_overall_figure(df_path=f"../../../spacer_scale_benchmarks/literature_correlations/BWRO_results.xlsx", y_lim="manual", y_lims_manual=y_lims_bwro)

