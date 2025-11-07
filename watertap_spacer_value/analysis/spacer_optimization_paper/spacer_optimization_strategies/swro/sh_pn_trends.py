import numpy as np
import pandas as pd
from pyomo.core import Objective, Var
from watertap_spacer_value.flowsheets.swro_single_stage import (
    build_swro_flowsheet as build_model,
)
from watertap_spacer_value.analysis.correlation_comparison.swro_single_stage import fix_model, scale_model, initialize_model
from watertap_spacer_value.flowsheets.ro_flowsheet_utils import (
    add_costing,
    degrees_of_freedom,
    solve,
    collect_micro_trend,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex



def compare_strategies():
    strategies = ["simulation", "topology", "module", "process", "topology_module", "module_process", "topology_process", "topology_module_process"]
    sh_pn_dfs = []
    macro_metrics = []
    for strategy in strategies:
        m = init_build_swro_flowsheet(nfe=60)

        for stage in m.fs.ro.values():
            add_telescoping_potential(stage)

        if not strategy == "simulation":
            add_lcow_objective(m)

        unfix_variables(m,  strategy)
        set_optimization_bounds(m)
        solve(m, tee=True, display=True)
        sh_pn_df = collect_micro_trend(m, keys=["Sh", "Pn", "Re"])
        sh_pn_dfs.append((strategy, sh_pn_df.copy()))
        macro_metrics.append({
            "Strategy": strategy,
            "Operating Pressure (bar)": m.fs.pump.outlet.pressure[0].value / 1e5,
            "SEC (kWh/m3)": m.fs.costing.specific_energy_consumption(),
            "LCOW ($/m3)": m.fs.costing.LCOW(),
            "Recovery (%)": m.fs.water_recovery.value * 100,
            "Sh factor": np.mean([stage.feed_side.Sh_improvement_factor.value for stage in m.fs.ro.values()]),
            "Friction factor": np.mean([stage.feed_side.friction_factor_improvement.value for stage in m.fs.ro.values()]),
            "Channel height (mm)": np.mean([stage.feed_side.channel_height.value * 1e3 for stage in m.fs.ro.values()]),
            "Spacer porosity": np.mean([stage.feed_side.spacer_porosity.value for stage in m.fs.ro.values()]),
            "Membrane length (m)": np.mean([stage.length.value for stage in m.fs.ro.values()]),
            "Pressure vessels": np.mean([stage.n_pressure_vessels.value for stage in m.fs.ro.values()]),
            "Inlet velocity (cm/s)": np.mean([stage.feed_side.velocity[0, 0].value * 1e2 for stage in m.fs.ro.values()]),
            "Membrane area (m2)": np.mean([stage.area.value for stage in m.fs.ro.values()]),
        })

    macro_df = pd.DataFrame(macro_metrics).set_index("Strategy")

    with pd.ExcelWriter("sh_pn_trends.xlsx") as writer:
        macro_df.to_excel(writer, sheet_name="Macro Metrics")
        for strategy, df in sh_pn_dfs:
            df.to_excel(writer, sheet_name=strategy)

    # Use cividis and convert to hex
    cmap = get_cmap("cividis")
    colors = [to_hex(cmap(i / len(strategies))) for i in range(len(strategies))]
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
        axs[1].plot(df["global_length_domain"], df["Re"], color=colors[idx], label=strategy, linewidth=2)
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
        Line2D([0], [0], color=colors[idx], lw=2, label=format_legend_label(strategy).replace("+", "+\n"))
        for idx, (strategy, _) in enumerate(sh_pn_dfs)
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.05),
        fontsize=12
    )
    plt.subplots_adjust(bottom=0.3, wspace=0.3, top=0.9)
    plt.savefig("sh_pn_trends_subplots.png", dpi=300)
    plt.show()

def unfix_variables(m, strategy):
    m.fs.water_recovery.fix(0.5)
    m.fs.pump.outlet.pressure[0].unfix()

    for stage in m.fs.ro.values():
        if strategy == "topology":
            stage.feed_side.Sh_improvement_factor.unfix()
            stage.feed_side.friction_factor_improvement.unfix()

        elif strategy == "module":
            stage.feed_side.channel_height.unfix()
            stage.feed_side.spacer_porosity.unfix()

        elif strategy == "process":
            stage.length.unfix()
            stage.n_pressure_vessels.unfix()

        elif strategy == "topology_module":
            stage.feed_side.Sh_improvement_factor.unfix()
            stage.feed_side.friction_factor_improvement.unfix()
            stage.feed_side.channel_height.unfix()
            stage.feed_side.spacer_porosity.unfix()

        elif strategy == "module_process":
            stage.feed_side.channel_height.unfix()
            stage.feed_side.spacer_porosity.unfix()
            stage.length.unfix()
            stage.n_pressure_vessels.unfix()

        elif strategy == "topology_process":
            stage.feed_side.Sh_improvement_factor.unfix()
            stage.feed_side.friction_factor_improvement.unfix()
            stage.length.unfix()
            stage.n_pressure_vessels.unfix()

        elif strategy == "topology_module_process":
            stage.feed_side.Sh_improvement_factor.unfix()
            stage.feed_side.friction_factor_improvement.unfix()
            stage.feed_side.channel_height.unfix()
            stage.feed_side.spacer_porosity.unfix()
            stage.length.unfix()
            stage.n_pressure_vessels.unfix()


    print(f"Degrees of freedom after using strategy: {degrees_of_freedom(m)}")
    return

def init_build_swro_flowsheet(nfe=60):
    m = build_model(correlation_type="schock", nfe=nfe)
    fix_model(m)
    scale_model(m)
    print(f"Degrees of freedom after fixing: {degrees_of_freedom(m)}")
    initialize_model(m, overpressure=2)
    print(f" Degrees of freedom after initialization: {degrees_of_freedom(m)}")
    solve(m, tee=False, display=False)
    print(f" Degrees of freedom after solving: {degrees_of_freedom(m)}")
    add_costing(m)
    print(f" Degrees of freedom after adding costing: {degrees_of_freedom(m)}")
    solve(m, tee=False, display=False)
    print(f" Degrees of freedom after solving with costing: {degrees_of_freedom(m)}")
    return m


def add_telescoping_potential(stage):
    stage.telescoping_potential = Var(
        initialize=1.0,
        bounds=(1e-3, 1.5),
        doc="Telescoping potential variable",
    )
    @stage.Constraint(
        doc = "Telescoping potential constraint",
    )
    def telescoping_potential_constraint(b):
        return b.telescoping_potential * b.length * 1e5 == -1 * b.deltaP[0] # Convert Pa to bar


def set_optimization_bounds(m):
    for stage in m.fs.ro.values():
        # Channel height between 0.3 mm and 1.5 mm
        stage.feed_side.channel_height.setlb(0.3e-3)
        stage.feed_side.channel_height.setub(1.5e-3)
        # Spacer porosity between 0.6 and 0.95
        stage.feed_side.spacer_porosity.setlb(0.6)
        stage.feed_side.spacer_porosity.setub(0.95)
        # Sh improvement factor between -90 and 90%
        stage.feed_side.Sh_improvement_factor.setlb(0.1)
        stage.feed_side.Sh_improvement_factor.setub(1.9)
        # Friction factor improvement between -90 and 90%
        stage.feed_side.friction_factor_improvement.setlb(0.1)
        stage.feed_side.friction_factor_improvement.setub(1.9)
        # Telescoping potential less than 1.5 to avoid excessive pressure drop
        stage.telescoping_potential.setub(1.5)
        stage.telescoping_potential.setlb(None)
        # Inlet velocity between 15 cm/s and 30 cm/s for optimal membrane performance, limits pressure vessel choice
        stage.feed_side.velocity[0, 0].setlb(0.15)
        stage.feed_side.velocity[0, 0].setub(0.30)
        # Membrane length between 1 m and 10 m to represent the elements in a pressure vessel
        stage.length.setlb(1.0)
        stage.length.setub(10.0)
        # Remove the bound on the width of the membrane to allow for optimization
        stage.feed_side.width.setub(None)


def format_legend_label(strategy):
    if strategy == "simulation":
        return "Fixed"

    parts = strategy.split("_")
    label_map = {
        "topology": "Spacer",
        "module": "Module",
        "process": "Process"
    }
    translated_parts = [label_map.get(p, p.capitalize()) for p in parts]
    return "+".join(translated_parts)



if __name__ == "__main__":
    compare_strategies()
