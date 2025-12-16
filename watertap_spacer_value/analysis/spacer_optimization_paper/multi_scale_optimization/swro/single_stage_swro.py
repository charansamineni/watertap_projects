import numpy as np
from watertap_spacer_value.analysis.spacer_optimization_paper.sensitivity_analysis.sweep_builder import (
    sweep_build,
)
from watertap_spacer_value.analysis.spacer_optimization_paper.spacer_scale_benchmarks.literature_correlations.RO_case_studies_across_correlations import (
    solve_for_recovery,
)
from watertap_spacer_value.flowsheets.ro_flowsheet_utils import *
from idaes.core.util.model_statistics import degrees_of_freedom
import pandas as pd
from itertools import combinations


def run_design_cases(
    nfe=10, velocity=0.20, ro_system="SWRO", salinity=35, correlation_type="schock"
):
    # Initialize dictionaries to store results
    micro_trends_df = {}
    macro_trends_df = {}
    # Create list of designs to evaluate
    design_scales = ["spacer", "module", "system"]
    # Generate design strategy combinations
    combinations_set = []
    for r in range(2, len(design_scales) + 1):
        for combo in combinations(design_scales, r):
            combinations_set.append("_".join(combo))
    # Add the single strategies and simulation
    possible_strategies = ["simulation"] + design_scales + combinations_set
    print(possible_strategies)
    for s in possible_strategies:
        # Build the model with required nfe and correlation
        m = sweep_build(
            nfe=nfe,
            velocity=velocity,
            ro_system=ro_system,
            salinity=salinity,
            correlation_type=correlation_type,
        )
        if s == "simulation":
            update_outputs(
                micro_trends_df=micro_trends_df,
                macro_trends_df=macro_trends_df,
                m=m,
                strategy=s,
            )
        # If system is not involved we can optimize the design variables as continuous variables
        elif "system" not in s:
            perform_cont_optimization_for_strategy(m, strategy=s)
            update_outputs(
                micro_trends_df=micro_trends_df,
                macro_trends_df=macro_trends_df,
                m=m,
                strategy=s,
            )
        else:  # System is involved, need to do grid search for l and n_pv
            perform_cont_optimization_for_strategy(m, strategy=s)
            l_opt = m.fs.ro[1].length.value
            n_pv_opt = m.fs.ro[1].n_pressure_vessels.value
            possible_l = [np.floor(l_opt), np.ceil(l_opt)]
            possible_pv = [np.floor(n_pv_opt), np.ceil(n_pv_opt)]
            grid_search_macros = []
            grid_search_micro_trends = {}
            for l in possible_l:
                for pv in possible_pv:
                    m.fs.ro[1].length.fix(l)
                    m.fs.ro[1].n_pressure_vessels.fix(pv)
                    try:
                        solve_for_recovery(m, recovery=0.5, tee=False, display=True, strategy="optimization")
                        macro_vars = collect_macro_variables(m)
                        populate_costs(m, target_list=macro_vars)
                        macro_vars.update({"l": l, "pv": pv})
                        grid_search_macros.append(macro_vars)
                        grid_search_micro_trends[f"{l}, {pv}"] = collect_micro_trend(m)
                    except Exception as e:
                        print(f"Skipping infeasible case l={l}, pv={pv}: {e}")

            grid_search_df = pd.DataFrame(grid_search_macros)
            best = min(grid_search_macros, key=lambda x: x["LCOW"])
            macro_trends_df[s] = {k: v for k, v in best.items() if k not in ["l", "pv"]}
            micro_trends_df[s] = grid_search_micro_trends[f"{best['l']}, {best['pv']}"]
            print(
                f"Completed strategy: {s} with optimal LCOW = {macro_trends_df[s]['LCOW']:.4f}"
            )
            grid_search_df.to_excel(
                f"grid_search_results_swro_{s}_{correlation_type}.xlsx", index=False
            )

    # Save the results to Excel
    macro_trends_df = pd.DataFrame(macro_trends_df).T
    print("\nMacro trends summary:")
    print(macro_trends_df.to_markdown(index=True))
    macro_trends_df["LCOW"] = pd.to_numeric(macro_trends_df["LCOW"], errors="coerce")
    opt_idx = macro_trends_df["LCOW"].idxmin()
    print(
        f"\nMost optimal design: {opt_idx} with LCOW = {macro_trends_df.loc[opt_idx, 'LCOW']:.4f}"
    )
    with pd.ExcelWriter(
        f"ro_optimization_strategies_results_swro_{correlation_type}.xlsx"
    ) as writer:
        for s, df in micro_trends_df.items():
            df.to_excel(writer, sheet_name=f"micro_{s}", index=False)
        macro_trends_df.to_excel(writer, sheet_name="macro_trends", index=True)
    return


def unfix_variables(m, strategy):
    print(f"Degrees of freedom before using {strategy}: {degrees_of_freedom(m)}")
    # Split strategy by underscores
    strategy_parts = strategy.split("_")

    for stage in m.fs.ro.values():
        for part in strategy_parts:
            if part in "spacer":
                stage.feed_side.Sh_improvement_factor.unfix()
                stage.feed_side.friction_factor_improvement.unfix()
            elif part in "module":
                stage.feed_side.channel_height.unfix()
                stage.feed_side.spacer_porosity.unfix()
            elif part in "system":
                stage.length.unfix()
                stage.n_pressure_vessels.unfix()
    print(f"Degrees of freedom after using {strategy}: {degrees_of_freedom(m)}")
    assert degrees_of_freedom(m) == 2 * len(m.fs.ro_stages) * len(
        strategy_parts
    ), "DOF is not correct after unfixing."
    return


def set_optimization_bounds(m):
    m.fs.pump.outlet.pressure[0].setub(85e5)  # 85 bar max for SWRO

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
        stage.feed_side.velocity[0, 0].setub(0.30)
        stage.feed_side.velocity[0, 0].setlb(0.15)
        # Membrane length between 1 m and 10 m to represent the elements in a pressure vessel
        stage.length.setlb(1.0)
        stage.length.setub(10.0)
        # Remove the bound on the width of the membrane to allow for optimization
        stage.feed_side.width.setub(None)


def collect_micro_trend(m):
    global_length_domain = []
    cumulative_length = 0.0
    # Build the global length domain
    for i in m.fs.ro_stages:
        stage = m.fs.ro[i]
        ld = list(stage.feed_side.length_domain)
        if len(ld) > 1:
            scaled_domain = [
                cumulative_length + x * value(stage.length)
                for x in (ld if i == 1 else ld[1:])
            ]
            global_length_domain.extend(scaled_domain)
            cumulative_length += value(stage.length)
    # Initialize dictionary for each key
    data = {"global_length_domain": global_length_domain}

    trend_dict = {}
    dens_solvent = m.fs.ro[1].dens_solvent.value
    for i in m.fs.ro_stages:
        stage = m.fs.ro[i]
        ld = list(stage.feed_side.length_domain)
        for x in ld if i == 1 else ld[1:]:
            trend_dict.setdefault("j_w", []).append(
                value(stage.flux_mass_phase_comp[0, x, "Liq", "H2O"]) / dens_solvent
                if x != 0
                else 0
            )
            trend_dict.setdefault("k_f", []).append(
                value(stage.feed_side.K[0, x, "TDS"])
            )
            trend_dict.setdefault("cp", []).append(
                value(stage.feed_side.cp_modulus[0, x, "TDS"])
            )
            trend_dict.setdefault("bulk_osmotic_pressure_bar", []).append(
                value(stage.feed_side.properties[0, x].pressure_osm_phase["Liq"]) / 1e5
            )
            trend_dict.setdefault("dp_dx_bar_per_m", []).append(
                value(stage.feed_side.dP_dx[0, x]) / 1e5
            )
            trend_dict.setdefault("operating_pressure_bar", []).append(
                value(stage.feed_side.properties[0, x].pressure) / 1e5
            )
            trend_dict.setdefault("velocity", []).append(
                value(stage.feed_side.velocity[0, x])
            )
            trend_dict.setdefault("f", []).append(
                value(stage.feed_side.friction_factor_darcy[0, x])
            )

    data.update(trend_dict)
    data["cp_penalty"] = [
        (cp - 1) * osm_p
        for cp, osm_p in zip(data["cp"], data["bulk_osmotic_pressure_bar"])
    ]
    dl = np.diff([0] + data["global_length_domain"])
    dp_dx = data["dp_dx_bar_per_m"]
    dp = [dx * dpdx for dx, dpdx in zip(dl, dp_dx)]
    total_dP = -1 * np.cumsum([0] + dp)
    data["dp_friction"] = total_dP[1:]  # Remove the initial zero
    data["jw_kf"] = [
        jw / kf if kf != 0 else 0 for jw, kf in zip(data["j_w"], data["k_f"])
    ]
    data["kinetic_energy"] = [0.5 * dens_solvent * v ** 2 for v in data["velocity"]]
    # Create DataFrame
    df = pd.DataFrame(data)
    return df


def update_outputs(micro_trends_df, macro_trends_df, m, strategy):
    micro_trends_df[strategy] = collect_micro_trend(m)
    macro_trends_df[strategy] = collect_macro_variables(m)
    populate_costs(target_list=macro_trends_df[strategy], m=m)

    print(
        f"Completed strategy: {strategy} with LCOW = {macro_trends_df[strategy]['LCOW']:.4f}"
    )
    print(f" Electricity cost : {m.fs.costing.electricity_cost.value} USD/kWh")


def populate_costs(m, target_list):
    total_mem_cost = sum(stage.costing.capital_cost.value for stage in m.fs.ro.values())
    levelized_mem_cost = (
        total_mem_cost
        * m.fs.costing.total_investment_factor.value
        * m.fs.costing.capital_recovery_factor.value
        / m.fs.costing.annual_water_production()
    )
    levelized_pump_cost = (
        m.fs.pump.costing.capital_cost.value
        * m.fs.costing.total_investment_factor.value
        * m.fs.costing.capital_recovery_factor.value
        / m.fs.costing.annual_water_production()
    )
    levelized_operating_cost = (
        m.fs.costing.total_operating_cost.value / m.fs.costing.annual_water_production()
    )
    target_list.update(
        {
            "levelized_mem_cost": levelized_mem_cost,
            "levelized_pump_cost": levelized_pump_cost,
            "levelized_operating_cost": levelized_operating_cost,
        }
    )


def perform_cont_optimization_for_strategy(m, strategy):
    if not hasattr(m.fs, "lcow_objective"):
        add_lcow_objective(m)
    unfix_variables(m, strategy)
    set_optimization_bounds(m)
    solution = solve_for_recovery(
        m, recovery=0.5, tee=False, display=True, strategy="optimization"
    )
    return solution


if __name__ == "__main__":

    for correlation in ["guillen", "schock", "dacosta", "koustou", "kuroda"]:
        run_design_cases(
            nfe=10,
            correlation_type=correlation,
            ro_system="SWRO",
            salinity=35,
            velocity=0.20,
        )
