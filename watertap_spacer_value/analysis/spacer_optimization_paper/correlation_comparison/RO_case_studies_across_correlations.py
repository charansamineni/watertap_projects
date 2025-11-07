import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mticker
from watertap.core.solvers import get_solver
from watertap_spacer_value.flowsheets.ro_flowsheet_utils import *
import pandas as pd
from pyomo.opt import assert_optimal_termination
from idaes.core.util.model_statistics import degrees_of_freedom


def plot_overall_figure(df_path="SWRO_results.xlsx", y_lim="auto", y_lims_manual=None):
    plt.rcParams.update({'font.size': 12, 'font.family': 'Arial'})

    # Read all sheets from the Excel file
    df_dict = pd.read_excel(df_path, sheet_name=None)

    # Create a figure with subplots
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.3)
    axes = [fig.add_subplot(gs[i]) for i in range(6)]

    # Plot configuration
    micro_plot_keys = ["K", "cp_modulus", "f", "deltaP"]
    ylabels = [
        "Mass transfer coefficient (m/s)",
        "CP modulus (-)",
        "Friction factor (-)",
        "Pressure drop (bar)"
    ]

    correlation_map = {
        "dacosta": {"color": "#4B0082", "legend": "DaCosta et al.", "marker": "o"},
        "guillen": {"color": "#008080", "legend": "Guillen et al.", "marker": "o"},
        "koustou": {"color": "#DAA520", "legend": "Koustou et al.", "marker": "o"},
        "kuroda": {"color": "#6A5ACD", "legend": "Kuroda et al.", "marker": "o"},
        "schock": {"color": "#DC143C", "legend": "Schock et al.", "marker": "o"},
    }

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
                        label=correlation_map[corr_key]["legend"],
                        marker=correlation_map[corr_key]["marker"],
                        s=25
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
                ax.annotate(stage, xy=(pos, 1.0), xycoords=("data", "axes fraction"),
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
            ax.set_yticks(np.linspace(yl_min, yl_max_rounded, 11))
        elif y_lim == "manual":
            ax.set_ylim(*y_lims_manual[idx])
            ax.set_yticks(np.linspace(*y_lims_manual[idx], 11))

        if key in ["K"]:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(sci_notation))
            ax.tick_params(axis='y', labelsize=12)

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
        [correlation_map[c]["legend"].replace(" et al.", "") for c in correlations],
        fontdict={'fontsize': 12, 'fontname': 'Arial'}
    )
    sec_ax.set_ylabel("Specific energy consumption (kWh/m³)", fontsize=12, fontname="Arial")

    if y_lim == "manual":
        sec_ax.set_ylim(*y_lims_manual[4])
        sec_ax.set_yticks(np.linspace(*y_lims_manual[4], 11))
    elif y_lim == "auto":
        yl_max = np.max(sec_vals)
        scale = 10 ** (np.floor(np.log10(yl_max)))
        yl_max_rounded = np.ceil(yl_max / scale * 10) * scale / 10
        sec_ax.set_ylim(0, yl_max_rounded)
        sec_ax.set_yticks(np.linspace(0, yl_max_rounded, 11))

    # LCOW Bar plot
    lcow_vals = [macro_df[macro_df["correlation"] == c]["LCOW"].values[0] for c in correlations]
    lcow_ax.bar(correlations,
                lcow_vals,
                edgecolor="black",
                linewidth=1,
                color=[correlation_map[c]["color"] for c in correlations])
    lcow_ax.set_xticks(range(len(correlations)))
    lcow_ax.set_xticklabels(
        [correlation_map[c]["legend"].replace(" et al.", "") for c in correlations],
        fontdict={'fontsize': 12, 'fontname': 'Arial'}
    )
    lcow_ax.set_ylabel("LCOW ($/m³)", fontsize=12, fontname="Arial")
    if y_lim == "manual":
        lcow_ax.set_ylim(*y_lims_manual[5])
        lcow_ax.set_yticks(np.linspace(*y_lims_manual[5], 11))
    elif y_lim == "auto":
        yl_max = np.max(lcow_vals)
        scale = 10 ** np.floor(np.log10(yl_max))
        yl_max_rounded = np.ceil(yl_max / scale) * scale
        lcow_ax.set_ylim(0, yl_max_rounded)
        lcow_ax.set_yticks(np.linspace(0, yl_max_rounded, 11))

    # Add legend
    fig.legend(
        handles=[
            plt.Line2D([0], [0],
                       color=correlation_map[c]["color"],
                       lw=2,
                       label=correlation_map[c]["legend"],
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
    plt.savefig(df_path.replace(".xlsx", ".svg"), format="svg", bbox_inches="tight", dpi=300, transparent=True)
    plt.show()


def run_case_studies(nfe=60, velocity=0.25, swro_salinity=35, bwro_salinity=5):
    swro_macro, swro_micro_dfs, swro_cv = run_analysis(
        nfe=nfe, velocity=velocity, ro_system="SWRO", salinity=swro_salinity
    )
    bwro_macro, bwro_micro_dfs, bwro_cv = run_analysis(
        nfe=nfe, velocity=velocity, ro_system="BWRO", salinity=bwro_salinity
    )

    # Create excels with the results
    with pd.ExcelWriter("SWRO_results.xlsx") as writer:
        pd.DataFrame(swro_macro).to_excel(writer, sheet_name="macro_results", index=False)
        for corr, df in swro_micro_dfs.items():
            df.to_excel(writer, sheet_name=f"{corr}_micro_trend", index=False)
        swro_cv.to_excel(writer, sheet_name="cv_results", index=False)
    with pd.ExcelWriter("BWRO_results.xlsx") as writer:
        pd.DataFrame(bwro_macro).to_excel(writer, sheet_name="macro_results", index=False)
        for corr, df in bwro_micro_dfs.items():
            df.to_excel(writer, sheet_name=f"{corr}_micro_trend", index=False)
        bwro_cv.to_excel(writer, sheet_name="cv_results", index=False)


def run_analysis(nfe=60, velocity=0.25, ro_system="SWRO", salinity=35):

    # Initialize lists to store results
    macro_results = []
    micro_trend_dfs = {}

    # Build the appropriate flowsheet based on the ro_system and initialize it
    for correlation in ["guillen", "schock", "dacosta", "koustou", "kuroda"]:
        print(f"Solving with {correlation} correlations")
        if ro_system.upper() == "SWRO":
            m = build_swro_flowsheet(correlation_type=correlation, nfe=nfe)
        elif ro_system.upper() == "BWRO":
            m = build_bwro_flowsheet(correlation_type=correlation, nfe=nfe)
        else:
            raise ValueError("ro_system must be either 'SWRO' or 'BWRO'")

        # Ensure osmotic pressures are initialized on the feed side
        for stage in m.fs.ro.values():
            for x in stage.feed_side.length_domain:
                stage.feed_side.properties[0, x].pressure_osm_phase[...]

        ro_system =  ro_system.upper()
        fix_model(m, velocity=velocity, salinity=salinity, ro_system=ro_system)
        scale_model(m, ro_system=ro_system)
        if ro_system == "BWRO":
            set_low_salinity_bounds(m)
        assert degrees_of_freedom(m) == 1, "Model is not fully specified after fixing."
        initialize_model(m, overpressure=2 if ro_system == "SWRO" else 7, ro_system=ro_system)
        assert degrees_of_freedom(m) == 0, "DOF is not zero after initialization."
        solve(m, tee=False, display=False)
        add_costing(m)
        solve(m, tee=False, display=False)
        solve_for_recovery(
            m, recovery=0.5 if ro_system == "SWRO" else 0.85, tee=False, display=True, strategy='simulation'
        )
        macro_vars = collect_macro_variables(m)
        macro_vars["correlation"] = correlation
        macro_vars["nfe"] = m.fs.ro[1].feed_side.nfe.value
        macro_vars["model"] = "swro_single_stage" if ro_system == "SWRO" else "bwro_321"
        macro_vars["solve"] = "recovery_0.5_simulation" if ro_system == "SWRO" else "recovery_0.85_simulation"
        macro_results.append(macro_vars)
        micro_trend_dfs[correlation] = collect_micro_trend(m)

    df = pd.DataFrame(macro_results)
    print(df.to_markdown(index=False))
    data_df = df.select_dtypes(include="number")
    cv = 100 * data_df.std(ddof=0) / data_df.mean()
    cv_df = cv.to_frame(name="CV (%)").reset_index().rename(columns={"index": "Variable"})
    return (
        macro_results,  # List of dictionaries with macro variables
        micro_trend_dfs,  # Dictionary of DataFrames with micro trend data
        cv_df, # Coefficient of Variation (CV) for numeric columns
    )


def solve_for_recovery(m, solver=None, tee=False, display=True, strategy='simulation', recovery=0.5):
    # Always ensure that bounds and constraints are applied before solving
    if strategy  == 'simulation':
        # Ensure LCOW objective is not active during simulation solves
        if hasattr(m.fs, "lcow_objective") and m.fs.lcow_objective.active:
            m.fs.lcow_objective.deactivate()
        # Unfix the pressure and fix the recovery as needed
        m.fs.pump.outlet.pressure[0].unfix()
        m.fs.water_recovery.fix(recovery)
        assert degrees_of_freedom(m) == 0, "DOF is not zero before simulation solve."

    elif strategy == 'optimization':
        if hasattr(m.fs, "lcow_objective") and not m.fs.lcow_objective.active:
            m.fs.lcow_objective.activate()
        elif not hasattr(m.fs, "lcow_objective"):
            m.fs.lcow_objective = Objective(
                expr=m.fs.costing.LCOW
            )

    # Solve the model
    results = solve(m, solver=solver, tee=tee, display=display)
    return results


def solve(m, solver=None, tee=False, display=True):
    if solver is None:
        solver = get_solver()
    results = solver.solve(m, tee=tee)
    assert_optimal_termination(results)
    if display:
        print_solved_state(m)
    return results


def sci_notation(x, pos):
    return f"{x:.1e}"



if __name__ == "__main__":
    # run_case_studies(nfe=60, velocity=0.25, swro_salinity=35, bwro_salinity=5)

    y_lims_swro = [
        (0, 2.5e-4),      # For K
        (1, 1.75),         # For cp_modulus
        (0, 3.5),   # For f
        (0, 5),          # For deltaP
        (0, 4.8),        # For SEC
        (0, 0.8)         # For LCOW
    ]
    y_lims_bwro = [
        (0, 2.5e-4),      # For K
        (1, 1.75),         # For cp_modulus
        (0, 3.5),   # For f
        (0, 5),         # For deltaP
        (0, 1.2),        # For SEC
        (0, 0.2)         # For LCOW
    ]

    plot_overall_figure(df_path="SWRO_results.xlsx", y_lim="manual", y_lims_manual=y_lims_swro)
    plot_overall_figure(df_path="BWRO_results.xlsx", y_lim="manual", y_lims_manual=y_lims_bwro)

    # plot_overall_figure(df_path="SWRO_results.xlsx", y_lim="auto")
    # plot_overall_figure(df_path="BWRO_results.xlsx", y_lim="auto")

