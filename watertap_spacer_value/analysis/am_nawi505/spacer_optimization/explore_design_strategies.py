import numpy as np
import pandas as pd
from pyomo.core import Objective, Var
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex
from sympy.physics.units import velocity

from watertap_spacer_value.flowsheets.aqua_membrane_flowsheet_utils import (
    build_am_swro_flowsheet,
    build_am_bwro_flowsheet,
    fix_aqua_membrane_case,
)
from watertap_spacer_value.flowsheets.ro_flowsheet_utils import (
    scale_model,
    initialize_model,
    print_solved_state,
    add_costing,
    add_lcow_objective,
    collect_micro_trend,
    collect_macro_variables,
)
from idaes.core.util.model_statistics import degrees_of_freedom
from watertap.core.solvers import get_solver


def run_overall_analysis():
    nfe = 10
    strategies = [
        "simulation",
        "topology",
        "module",
        "process",
        "module_process",
    ]
    v = 0.25  # m/s
    h = 0.5e-3  # m
    p = 0.99  # porosity
    n = 180  # number of vessels
    for system in ["SWRO", "BWRO"]:
        compare_strategies(
            ro_system=system,
            strategies=strategies,
            nfe=nfe,
            display=False,
            velocity=v,
            channel_height=h,
            porosity=p,
            n_vessels=n,
        )

def compare_strategies(
    ro_system,
    strategies=None,
    nfe=10,
    display=False,
    velocity=0.25,
    channel_height=0.5e-3,
    porosity=0.95,
    n_vessels=180,
):
    if strategies is None:
        strategies = [
            "simulation",
        ]
    sh_pn_dfs = []
    macro_metrics = []
    for strategy in strategies:
        m = init_build_ro_flowsheet(
            ro_system=ro_system,
            nfe=nfe,
            display=display,
            velocity=velocity,
            channel_height=channel_height,
            porosity=porosity,
            n_vessels=n_vessels,
        )

        for stage in m.fs.ro.values():
            add_telescoping_potential(stage)

        if not strategy == "simulation":
            add_lcow_objective(m)
        unfix_variables(m, strategy)

        # Unfix the pressure and fix recovery to allow for optimization
        if ro_system == "SWRO":
            m.fs.pump.outlet.pressure[0].unfix()
            m.fs.water_recovery.fix(0.5)

        elif ro_system == "BWRO":
            m.fs.pump.outlet.pressure[0].unfix()
            m.fs.water_recovery.fix(0.85)
        else:
            raise ValueError("ro_system must be 'SWRO' or 'BWRO'")

        set_optimization_bounds(m)
        solve(m, tee=True, display=True)
        metrics = collect_macro_variables(m)
        macro_metrics.append({"Strategy": format_legend_label(strategy), **metrics})
        sh_pn_df = collect_micro_trend(m)
        sh_pn_dfs.append((format_legend_label(strategy), sh_pn_df))

    macro_df = pd.DataFrame(macro_metrics).set_index("Strategy")

    # Calculate savings and add as columns
    if "LCOW" in macro_df.columns and "Simulation" in macro_df.index:
        base_lcow = macro_df.loc["Simulation", "LCOW"]
        macro_df["Savings_vs_Simulation_%"] = (
            (base_lcow - macro_df["LCOW"]) / base_lcow * 100
        )
        macro_df["Base_LCOW"] = base_lcow

    print(macro_df)

    # Write macro_df and each micro_trend DataFrame to separate sheets in one Excel file
    with pd.ExcelWriter(f"sh_pn_trends_{ro_system.lower()}.xlsx") as writer:
        macro_df.to_excel(writer, sheet_name="Macro Metrics")
        for strategy, df in sh_pn_dfs:
            df.to_excel(writer, sheet_name=strategy)


def init_build_ro_flowsheet(
    ro_system="SWRO",
    nfe=10,
    display=False,
    velocity=0.25,
    channel_height=0.5e-3,
    porosity=0.95,
    n_vessels=180,
):
    if ro_system == "SWRO":
        m = build_am_swro_flowsheet(nfe=nfe)
        salinity = 35
        recovery_target = 0.5
        over_pressure = 3
    else:
        m = build_am_bwro_flowsheet(nfe=nfe)
        salinity = 5
        over_pressure = 8
        recovery_target = 0.85
    fix_aqua_membrane_case(
        m,
        velocity=velocity,
        salinity=salinity,
        channel_height=channel_height,
        porosity=porosity,
        n_vessels=n_vessels,
        ro_system=ro_system,
    )
    scale_model(m, ro_system=ro_system)
    initialize_model(m, ro_system=ro_system, overpressure=over_pressure)
    add_costing(m)
    solve(m, tee=False, display=False)
    m.fs.water_recovery.fix(recovery_target)
    m.fs.pump.outlet.pressure[0].unfix()
    results = solve(m, tee=False, display=display)
    return m


def solve(m, tee=True, display=True):
    solver = get_solver()
    results = solver.solve(m, tee=tee)
    assert str(results.solver.termination_condition) == "optimal"
    if display:
        print_solved_state(m)
    return results


def micro_trend_plot(sh_pn_dfs):
    # Use cividis and convert to hex
    cmap = get_cmap("cividis")
    colors = [to_hex(cmap(i)) for i in np.linspace(0, 1, len(sh_pn_dfs))]
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 14

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    axs = axs.flatten()

    # Sherwood Number vs Power Number plot
    for idx, (strategy, df) in enumerate(sh_pn_dfs):
        axs[0].plot(df["Sh"], df["Pn"], color=colors[idx], label=strategy, linewidth=2)

    axs[0].set_xscale("log")
    axs[0].set_xlim(1, 100)
    axs[0].set_yscale("log")
    axs[0].set_ylim(1e4, 1e7)

    # Major ticks only at decades (10, 100, etc.)
    axs[0].xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
    axs[0].xaxis.set_major_formatter(LogFormatterMathtext())
    axs[0].xaxis.set_minor_locator(plt.NullLocator())  # Disable minor ticks

    axs[0].yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
    axs[0].yaxis.set_major_formatter(LogFormatterMathtext())
    axs[0].yaxis.set_minor_locator(plt.NullLocator())  # Disable minor ticks

    axs[0].set_xlabel("Sherwood Number (-)")
    axs[0].set_ylabel("Power Number (-)")
    axs[0].set_title("Sherwood Number vs Power Number")
    axs[0].grid(False)

    # 2. Length domain vs Re
    for idx, (strategy, df) in enumerate(sh_pn_dfs):
        axs[1].plot(
            df["global_length_domain"],
            df["Re"],
            color=colors[idx],
            label=strategy,
            linewidth=2,
        )
    axs[1].set_xlabel("Position along RO train (m)")
    axs[1].set_xscale("linear")
    axs[1].set_xlim(0, 10)
    axs[1].set_xticks(np.linspace(0, 10, 11))
    axs[1].set_ylabel("Reynolds Number (-)")
    axs[1].set_yscale("linear")
    axs[1].set_ylim(0, 250)
    axs[1].set_yticks(np.linspace(0, 250, 11))
    axs[1].set_title("Membrane Length vs Reynolds Number")
    axs[1].grid(False)

    # Custom legend below all subplots
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=colors[idx],
            lw=2,
            label=format_legend_label(strategy).replace("+", "+\n"),
        )
        for idx, (strategy, _) in enumerate(sh_pn_dfs)
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.05),
        fontsize=12,
    )
    plt.subplots_adjust(bottom=0.3, wspace=0.3, top=0.9)
    plt.savefig("sh_pn_trends_subplots.png", dpi=300)
    plt.show()


def unfix_variables(m, strategy):

    for stage in m.fs.ro.values():
        if strategy == "topology":
            stage.feed_side.k_error_factor.unfix()
            stage.feed_side.f_error_factor.unfix()

        elif strategy == "module":
            stage.feed_side.channel_height.unfix()
            stage.feed_side.spacer_porosity.unfix()

        elif strategy == "process":
            stage.length.unfix()
            stage.n_pressure_vessels.unfix()

        elif strategy == "topology_module":
            stage.feed_side.k_error_factor.unfix()
            stage.feed_side.f_error_factor.unfix()
            stage.feed_side.channel_height.unfix()
            stage.feed_side.spacer_porosity.unfix()

        elif strategy == "module_process":
            stage.feed_side.channel_height.unfix()
            stage.feed_side.spacer_porosity.unfix()
            stage.length.unfix()
            stage.n_pressure_vessels.unfix()

        elif strategy == "topology_process":
            stage.feed_side.k_error_factor.unfix()
            stage.feed_side.f_error_factor.unfix()
            stage.length.unfix()
            stage.n_pressure_vessels.unfix()

        elif strategy == "topology_module_process":
            stage.feed_side.k_error_factor.unfix()
            stage.feed_side.f_error_factor.unfix()
            stage.feed_side.channel_height.unfix()
            stage.feed_side.spacer_porosity.unfix()
            stage.length.unfix()
            stage.n_pressure_vessels.unfix()

    print(f"Degrees of freedom after using strategy {str(strategy)}: {degrees_of_freedom(m)}")
    return


def add_telescoping_potential(stage):
    stage.telescoping_potential = Var(
        initialize=1.0, bounds=(1e-3, 1.5), doc="Telescoping potential variable",
    )

    @stage.Constraint(doc="Telescoping potential constraint",)
    def telescoping_potential_constraint(b):
        return (
            b.telescoping_potential * b.length * 1e5 == -1 * b.deltaP[0]
        )  # Convert Pa to bar


def set_optimization_bounds(m):
    for stage in m.fs.ro.values():
        # Channel height between 0.3 mm and 1.5 mm
        stage.feed_side.channel_height.setlb(0.15e-3)
        stage.feed_side.channel_height.setub(1.5e-3)
        # Spacer porosity between 0.6 and 0.95
        stage.feed_side.spacer_porosity.setlb(0.6)
        stage.feed_side.spacer_porosity.setub(1)
        # Sh improvement factor between -90 and 90%
        stage.feed_side.k_error_factor.setlb(0.5)
        stage.feed_side.k_error_factor.setub(1.5)
        # Friction factor improvement between -90 and 90%
        stage.feed_side.f_error_factor.setlb(0.5)
        stage.feed_side.f_error_factor.setub(1.5)
        # Telescoping potential less than 1.5 to avoid excessive pressure drop
        stage.telescoping_potential.setub(1.5)
        stage.telescoping_potential.setlb(None)
        if stage == m.fs.ro[1]:
            # Inlet velocity between 15 cm/s and 30 cm/s for optimal membrane performance, limits pressure vessel choice
            stage.feed_side.velocity[0, 0].setlb(0.15)
            stage.feed_side.velocity[0, 0].setub(0.30)
        # Membrane length between 1 m and 10 m to represent the elements in a pressure vessel
        stage.length.setlb(1.0)
        stage.length.setub(10.0)
        # Remove the bound on the width of the membrane to allow for optimization
        stage.feed_side.width.setub(None)


def format_legend_label(strategy):
    mapping = {
        "topology": "Spacer",
        "module": "Feed channel",
        "process": "Process",
        "topology_module": "Spacer+Feed channel",
        "module_process": "Feed channel+Process",
        "topology_process": "Spacer+Process",
        "topology_module_process": "Spacer+Feed channel+Process",
        "simulation": "Simulation",
    }
    return mapping.get(strategy, strategy.capitalize())


if __name__ == "__main__":
    run_overall_analysis()
