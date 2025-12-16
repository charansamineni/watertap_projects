import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib import gridspec
from watertap_spacer_value.plotting_tools.paul_tol_color_maps import gen_paultol_colormap
from watertap_spacer_value.analysis.spacer_optimization_paper.data_analysis.cost_savings_opportunities.overall_cost_savings_opportunities import set_plot_style
from matplotlib import cm

def get_strategy_map():
    colors = gen_paultol_colormap("tol_vibrant", 6)

    return {
        "simulation": {
            "color": colors[0],
            "marker": "o",
            "linestyle": "--",
            "label": "Base case",
        },
        "spacer": {
            "color": colors[1],
            "marker": "o",
            "linestyle": "--",
            "label": "Spacer",
        },
        "module": {
            "color": colors[2],
            "marker": "o",
            "linestyle": "--",
            "label": "Module",
        },
        "system": {
            "color": colors[3],
            "marker": "o",
            "linestyle": "--",
            "label": "System",
        },
        "spacer_module_system": {
            "color": colors[4],
            "marker": "o",
            "linestyle": "--",
            "label": "Combined",
        },
    }



def _plot_line(ax, x, y, style):
    """Small helper to standardize line plotting."""
    return ax.plot(
        x, y,
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
        linewidth=2,
        markersize=5,
    )



def plot_multiple_strategies(strategies_to_plot=None, data_file=None):

    strategies_to_plot = set(strategies_to_plot or [])

    strategy_map = get_strategy_map()
    all_strategies = ["simulation", "spacer", "module", "system", "spacer_module_system"]
    strategy_labels = [strategy_map[s]["label"] for s in all_strategies]

    # --- Load results ---
    if not data_file:
        data_file = "../../multi_scale_optimization/bwro/ro_optimization_strategies_results_bwro_schock.xlsx"
    xls = pd.ExcelFile(data_file)

    results = {}
    macro_results = pd.read_excel(xls, sheet_name="macro_trends", index_col=0)

    for sheet_name in xls.sheet_names:
        if not sheet_name.startswith("micro_"):
            continue

        strategy = sheet_name.replace("micro_", "")
        if strategies_to_plot and strategy not in strategies_to_plot:
            continue

        df = pd.read_excel(xls, sheet_name=sheet_name)
        lcow = macro_results.loc[strategy, "LCOW"]
        results[strategy] = (df, lcow)

    # --- Plot setup ---
    set_plot_style()
    font = {"family": "Arial", "size": 12}


    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.4, hspace=0.4)
    axs = np.array([fig.add_subplot(gs[i]) for i in range(9)]).reshape(3, 3)

    for ax in axs.flat:
        ax.set_box_aspect(1)
        ax.tick_params(axis="both", labelsize=12)

    handles_labels = []

    # --- Micro-scale plots ---
    for strategy, (df, lcow) in results.items():
        style = strategy_map[strategy]

        print(f"\nStrategy: {strategy} (LCOW={lcow:.3f})")
        for col in ['j_w', 'k_f', 'cp_penalty', 'kinetic_energy', 'f', 'dp_friction']:
            if col not in df:
                continue
            data = df[col] if col == "kinetic_energy" else df[col].iloc[1:]
            print(f"{col}: min={data.min():.3g}, max={data.max():.3g}")

        label_base = strategy_map[strategy]["label"]

        label = (
            f"{label_base} optimization (LCOW={lcow:.2f} $/m³)"
            if strategy != "simulation"
            else f"{label_base} (LCOW={lcow:.2f} $/m³)"
        )

        h0, = _plot_line(axs[0, 0], df['global_length_domain'][1:], df['j_w'][1:], style)
        _plot_line(axs[0, 1], df['global_length_domain'][1:], df['k_f'][1:], style)
        _plot_line(axs[0, 2], df['global_length_domain'][1:], df['cp_penalty'][1:], style)
        _plot_line(axs[1, 0], df['global_length_domain'], df['kinetic_energy'], style)
        _plot_line(axs[1, 1], df['global_length_domain'][1:], df['f'][1:], style)
        _plot_line(axs[1, 2], df['global_length_domain'][1:], df['dp_friction'][1:], style)

        handles_labels.append((h0, label))

    # --- Macro bar plots ---
    macro_strategies = [s for s in strategies_to_plot if s in results]
    macro_df = macro_results.loc[macro_strategies]
    macro_colors = [strategy_map[s]["color"] for s in macro_strategies]

    all_strategies = ["simulation", "spacer", "module", "system", "spacer_module_system"]

    # Membrane area
    axs[2, 0].bar(
        range(5),
        [macro_df["Total membrane area"].get(s, 0) for s in all_strategies],
        color=[strategy_map.get(s, {}).get("color", "#eeeeee") for s in all_strategies],
        edgecolor="black",
    )

    # Operating pressure
    axs[2, 1].bar(
        range(5),
        [macro_df["Operating pressure"].get(s, 0) / 1e5 for s in all_strategies],
        color=[strategy_map.get(s, {}).get("color", "#eeeeee") for s in all_strategies],
        edgecolor="black",
    )


    # Cost breakdown
    cost_keys = ["levelized_mem_cost", "levelized_pump_cost", "levelized_operating_cost", "LCOW"]
    legend_labels = ["Membrane CapEX", "Pump CapEX", "OpEX", "LCOW"]
    hatch_patterns = ['|', '\\', '/', '']
    bar_width = 0.18

    bax = axs[2, 2]
    macro_x = np.array([all_strategies.index(s) for s in macro_strategies])

    for i, (key, hatch) in enumerate(zip(cost_keys, hatch_patterns)):
        bax.bar(
            macro_x + i * bar_width,
            macro_df[key],
            width=bar_width,
            color=macro_colors,
            edgecolor="black",
            hatch=hatch,
        )

    # Optional: add hatch legend
    legend_patches = [
        mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch_patterns[i], label=legend_labels[i])
        for i in range(len(legend_labels))]
    bax.legend(handles=legend_patches, loc='upper right', fontsize=10, ncol=2, frameon=False)
    # --- Legend ---
    handles, labels = zip(*handles_labels)
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        prop=font,
        bbox_to_anchor=(0.5, 0.02),
    )

    for ax in np.concatenate((axs[0, :], axs[1, :])):
        if 'bwro' in data_file:
            ax.set_xlim(0, 21)
            ax.set_xticks(np.linspace(0, 21, 8))
            ax.set_xlabel(
                "Axial position along the RO system (m)", fontdict=font
            )
        elif 'swro' in data_file:
            ax.set_xlim(0, 10)
            ax.set_xticks(np.linspace(0, 10, 11))
            ax.set_xlabel(
                "Axial position along the RO system (m)", fontdict=font
            )

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))


    axs[0, 0].set_ylabel("Water flux (m/s)", fontdict=font)
    axs[0, 0].yaxis.set_major_formatter(formatter)
    axs[0, 0].set_yticks(np.linspace(0, 2.5e-5, 6))
    axs[0, 0].set_ylim(0, 2.5e-5)

    # Mass transfer coefficient
    axs[0, 1].set_ylabel("Mass transfer coefficient (m/s)", fontdict=font)
    axs[0, 1].yaxis.set_major_formatter(formatter)
    axs[0, 1].set_yticks(np.linspace(0, 2.5e-4, 6))
    axs[0, 1].set_ylim(0, 2.5e-4)

    # Membrane area (macro)
    axs[2, 0].set_ylabel("Membrane area (m$^2$)", fontdict=font)
    axs[2, 0].yaxis.set_major_formatter(formatter)
    axs[2, 0].set_yticks(np.linspace(0, 1.5e5, 6))
    axs[2, 0].set_ylim(0, 1.5e5)
    axs[2, 0].set_xticks(range(5))
    axs[2, 0].set_xticklabels(
        strategy_labels,
        fontdict=font,
        rotation=30,
        ha="right",
    )

    # Cost breakdown (macro)
    axs[2, 2].set_ylabel("Cost (USD/m$^3$)", fontdict=font)
    if 'bwro' in data_file:
        axs[2, 2].set_yticks(np.linspace(0, 0.5, 6))
        axs[2, 2].set_ylim(0, 0.5)
    elif 'swro' in data_file:
        axs[2, 2].set_yticks(np.linspace(0, 1.5, 6))
        axs[2, 2].set_ylim(0, 1.5)
    axs[2, 2].set_xticks(range(5))
    axs[2, 2].set_xticklabels(
        strategy_labels,
        fontdict=font,
        rotation=30,
        ha="right",
    )


    # Operating pressure (macro)
    axs[2, 1].set_ylabel("Operating pressure (bar)", fontdict=font)
    if 'bwro' in data_file:
        axs[2, 1].set_yticks(np.linspace(0, 30, 6))
        axs[2, 1].set_ylim(0, 30)
    elif 'swro' in data_file:
        axs[2, 1].set_yticks(np.linspace(0, 100, 6))
        axs[2, 1].set_ylim(0, 100)
    axs[2, 1].set_xticks(range(5))
    axs[2, 1].set_xticklabels(
        strategy_labels,
        fontdict=font,
        rotation=30,
        ha="right",
    )

    # Osmotic back pressure
    axs[0, 2].set_ylabel("Osmotic back pressure (bar)", fontdict=font)
    axs[0, 2].set_yticks(np.linspace(0, 15, 6))
    axs[0, 2].set_ylim(0, 15)

    # Dynamic pressure
    axs[1, 0].set_ylabel(r"Dynamic pressure (Pa) ($\frac{1}{2} \rho u^2$)", fontdict=font)
    axs[1, 0].set_yticks(np.linspace(0, 50, 6))
    axs[1, 0].set_ylim(0, 50)

    # Friction factor
    axs[1, 1].set_ylabel("Friction factor", fontdict=font)
    axs[1, 1].set_yticks(np.linspace(0, 2.5, 6))
    axs[1, 1].set_ylim(0, 2.5)

    # Spacer induced pressure drop
    axs[1, 2].set_ylabel("Spacer induced pressure drop (bar)", fontdict=font)
    axs[1, 2].set_yticks(np.linspace(0, 7.5, 6))
    axs[1, 2].set_ylim(0, 7.5)

    plt.subplots_adjust(bottom=0.15)
    file_suffix = data_file .split("/")[-1].replace("ro_optimization_strategies_results_", "").replace(".xlsx", "")
    print(f"Writing {file_suffix}...")
    plt.savefig(
        f'{file_suffix}_optimization_trends.svg',
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_normalized_micro_strategies(strategies_to_plot=None, correlation=None):
    strategies_to_plot = set(strategies_to_plot or [])
    if correlation is None:
        correlation = "schock"

    files = {
        "SWRO": f"../../multi_scale_optimization/swro/ro_optimization_strategies_results_swro_{correlation}.xlsx",
        "BWRO": f"../../multi_scale_optimization/bwro/ro_optimization_strategies_results_bwro_{correlation}.xlsx",
    }

    # Strategy map for colors/labels
    strategy_map = get_strategy_map()

    # Variables to normalize and plot
    columns_to_plot = ["bulk_osmotic_pressure_bar", "cp_penalty", "dp_friction", "operating_pressure_bar"]
    ylabels = [
        "Osmotic pressure",
        "Osmotic back pressure",
        "Friction losses",
        "Operating pressure",
    ]
    var_colors = gen_paultol_colormap("tol_muted", len(columns_to_plot))
    line_styles = ["-", "--", ":", "-."]

    set_plot_style()
    font = {"family": "Arial", "size": 12}

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 1, hspace=0.4, wspace=0.4)
    axs = [fig.add_subplot(gs[i]) for i in range(2)]


    for ax, (label, data_file) in zip(axs, files.items()):
        xls = pd.ExcelFile(data_file)
        results = {}
        max_op_pressure = 0.0

        # --- Load all micro sheets ---
        for sheet_name in xls.sheet_names:
            if not sheet_name.startswith("micro_"):
                continue
            strategy = sheet_name.replace("micro_", "")
            if strategies_to_plot and strategy not in strategies_to_plot:
                continue
            df = pd.read_excel(xls, sheet_name=sheet_name)
            df.columns = [c.lower() for c in df.columns]  # lowercase columns
            results[strategy] = df
            max_op_pressure = max(max_op_pressure, df.get("operating_pressure_bar", pd.Series([0])).max())

        if not results:
            continue

        # --- Plot variables ---
        for s_idx, (strategy, df) in enumerate(results.items()):
            ls = line_styles[s_idx % len(line_styles)]
            style = strategy_map.get(strategy, {"label": strategy})
            for v_idx, col in enumerate(columns_to_plot):
                if col not in df:
                    continue
                denom = max_op_pressure if max_op_pressure > 0 else df[col].max()
                denom = denom if denom != 0 else 1.0
                ydata = df[col] / denom
                ax.plot(
                    df["global_length_domain"],
                    ydata,
                    color=var_colors[v_idx],
                    linestyle=ls,
                    linewidth=2,
                    marker="o",
                    markersize = 5,
                    label=f"{ylabels[v_idx]}",
                )

        ax.set_title(label, fontdict=font)
        ax.set_xlabel("Axial position along the RO system (m)", fontdict=font)
        ax.set_ylim(-0.25, 1.25)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.legend(
            loc="best",
            fontsize=8,
            frameon=False,
        )

    axs[0].set_xticks(np.linspace(0, 6, 7))
    axs[0].set_xlim(0, 6)
    axs[1].set_xticks(np.linspace(0, 18, 7))
    axs[1].set_xlim(0, 18)


    # Shared y-axis label
    axs[0].set_ylabel("Normalized with operating pressure", fontdict=font)
    axs[0].set_ylabel("Normalized with operating pressure", fontdict=font)

    print(f"Writing normalized_micro_strategies_{correlation}...")
    plt.savefig(
        f'normalized_micro_strategies_{correlation}.svg',
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()



def plot_normalized_micro_strategies_v2(strategies_to_plot=None, correlation=None):
    strategies_to_plot = set(strategies_to_plot or [])
    if correlation is None:
        correlation = "schock"

    files = {
        "SWRO": f"../../multi_scale_optimization/swro/ro_optimization_strategies_results_swro_{correlation}.xlsx",
        "BWRO": f"../../multi_scale_optimization/bwro/ro_optimization_strategies_results_bwro_{correlation}.xlsx",
    }

    # Strategy map for colors/labels
    strategy_map = get_strategy_map()

    # Variables to normalize and plot
    columns_to_plot = ["bulk_osmotic_pressure_bar", "cp_penalty", "dp_friction", "operating_pressure_bar"]
    ylabels = [
        "Osmotic pressure",
        "Osmotic back pressure",
        "Friction losses",
        "Operating pressure",
    ]
    var_colors = gen_paultol_colormap("tol_muted", len(columns_to_plot))
    line_styles = ["-", "--", ":", "-."]

    set_plot_style()
    font = {"family": "Arial", "size": 12}

    # --- Normalized line plots ---
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 1, hspace=0.4, wspace=0.4)
    axs = [fig.add_subplot(gs[i]) for i in range(2)]

    for ax, (label, data_file) in zip(axs, files.items()):
        xls = pd.ExcelFile(data_file)
        results = {}
        max_op_pressure = 0.0

        # --- Load all micro sheets ---
        for sheet_name in xls.sheet_names:
            if not sheet_name.startswith("micro_"):
                continue
            strategy = sheet_name.replace("micro_", "")
            if strategies_to_plot and strategy not in strategies_to_plot:
                continue
            df = pd.read_excel(xls, sheet_name=sheet_name)
            df.columns = [c.lower() for c in df.columns]  # lowercase columns
            results[strategy] = df
            max_op_pressure = max(max_op_pressure, df.get("operating_pressure_bar", pd.Series([0])).max())

        if not results:
            continue

        # --- Plot variables ---
        for s_idx, (strategy, df) in enumerate(results.items()):
            ls = line_styles[s_idx % len(line_styles)]
            for v_idx, col in enumerate(columns_to_plot):
                if col not in df:
                    continue
                denom = max_op_pressure if max_op_pressure > 0 else df[col].max()
                denom = denom if denom != 0 else 1.0
                ydata = df[col] / denom
                ax.plot(
                    df["global_length_domain"],
                    ydata,
                    color=var_colors[v_idx],
                    linestyle=ls,
                    linewidth=2,
                    marker="o",
                    markersize=5,
                    label=f"{ylabels[v_idx]}",
                )

        ax.set_title(label, fontdict=font)
        ax.set_xlabel("Axial position along the RO system (m)", fontdict=font)
        ax.set_ylim(-0.25, 1.25)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.legend(
            loc="best",
            fontsize=8,
            frameon=False,
        )

    axs[0].set_xticks(np.linspace(0, 6, 7))
    axs[0].set_xlim(0, 6)
    axs[1].set_xticks(np.linspace(0, 18, 7))
    axs[1].set_xlim(0, 18)

    # Shared y-axis label
    axs[0].set_ylabel("Normalized with operating pressure", fontdict=font)

    print(f"Writing normalized_micro_strategies_{correlation}...")
    plt.savefig(
        f'normalized_micro_strategies_{correlation}.svg',
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # --- Grouped bar plots: average values per strategy for each file ---
    for label, data_file in files.items():
        xls = pd.ExcelFile(data_file)
        means = {}
        for sheet_name in xls.sheet_names:
            if not sheet_name.startswith("micro_"):
                continue
            strategy = sheet_name.replace("micro_", "")
            if strategies_to_plot and strategy not in strategies_to_plot:
                continue
            df = pd.read_excel(xls, sheet_name=sheet_name)
            df.columns = [c.lower() for c in df.columns]
            vals = {}
            for col in columns_to_plot:
                vals[col] = df[col].dropna().mean() if col in df else 0.0
            means[strategy] = vals

        if not means:
            continue

        strategies = list(means.keys())
        n = len(strategies)
        ind = np.arange(n)
        width = 0.18

        fig, ax = plt.subplots(figsize=(max(6, n * 1.2), 4))
        for i, col in enumerate(columns_to_plot):
            data = [means[s][col] for s in strategies]
            ax.bar(ind + i * width, data, width, label=ylabels[i], color=var_colors[i], edgecolor="black")

        ax.set_xticks(ind + width * (len(columns_to_plot) - 1) / 2)
        labels = [strategy_map.get(s, {}).get("label", s) for s in strategies]
        ax.set_xticklabels(labels, rotation=30, ha="right", fontdict=font)
        ax.set_ylabel("Average value", fontdict=font)
        ax.set_title(f"Average micro variables - {label}", fontdict=font)
        ax.legend(frameon=False, fontsize=9)
        plt.tight_layout()
        out_name = f'grouped_avg_micro_strategies_{label.lower()}_{correlation}.svg'
        print(f"Writing {out_name}...")
        plt.savefig(out_name, dpi=300, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    strategies = [
        "simulation",
        "spacer",
        "module",
        "system",
        "spacer_module_system",
    ]
    for data_file in [
        "../../multi_scale_optimization/swro/ro_optimization_strategies_results_swro_schock.xlsx",
        "../../multi_scale_optimization/bwro/ro_optimization_strategies_results_bwro_schock.xlsx",
    ]:
        plot_multiple_strategies(strategies_to_plot=strategies, data_file=data_file)

    # plot_normalized_micro_strategies_v2(
    #     strategies_to_plot=[
    #         "simulation",
    #         # "spacer", "module", "system",
    #         # "spacer_system", "spacer_module",
    #         # "module_system",
    #         # "spacer_module_system",
    #     ],
    #     correlation="schock"
    # )
