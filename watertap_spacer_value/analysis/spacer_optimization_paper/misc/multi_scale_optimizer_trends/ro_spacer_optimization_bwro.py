import numpy as np
from watertap_spacer_value.flowsheets.ro_flowsheet_utils import *
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.opt import assert_optimal_termination
from watertap.core.solvers import get_solver
from watertap_spacer_value.analysis.spacer_optimization_paper.spacer_scale_benchmarks.literature_correlations.RO_case_studies_across_correlations import sci_notation
import pandas as pd
from pyomo.environ import Var

def run_multiple_strategies(strategies, nfe=10, correlation_type="schock"):
    micro_trends_df = {}
    macro_trends_df = {}

    for s in strategies:
        if s not in [
            "simulation",
            "topology",
            "module",
            "process",
            "topology_module",
            "module_process",
            "topology_process",
            "topology_module_process"
        ]:
            raise ValueError(f"Strategy {s} is not recognized. Valid strategies are: "
                             "'simulation', 'topology', 'module', 'process', "
                             "'topology_module', 'module_process', 'topology_process', "
                             "'topology_module_process'.")



        # Build the model with required nfe and correlation
        m = init_build_bwro_flowsheet(nfe=nfe, correlation_type=correlation_type)

        if s == "simulation":
            unfix_variables(m, s)
            set_optimization_bounds(m)
            solve_for_recovery(m, recovery=0.85, tee=False, display=True)
            micro_trends_df[s] = collect_micro_trend(m)
            macro_trends_df[s] = collect_macro_variables(m)

            total_mem_cost = sum(stage.costing.capital_cost.value for stage in m.fs.ro.values())
            levelized_mem_cost = total_mem_cost * m.fs.costing.total_investment_factor.value * m.fs.costing.capital_recovery_factor.value / m.fs.costing.annual_water_production()
            levelized_pump_cost = m.fs.pump.costing.capital_cost.value * m.fs.costing.total_investment_factor.value * m.fs.costing.capital_recovery_factor.value / m.fs.costing.annual_water_production()
            levelized_operating_cost = m.fs.costing.total_operating_cost.value / m.fs.costing.annual_water_production()
            macro_trends_df[s].update({
                "levelized_mem_cost": levelized_mem_cost,
                "levelized_pump_cost": levelized_pump_cost,
                "levelized_operating_cost": levelized_operating_cost
            })
            print(f"Completed strategy: {s} with LCOW = {macro_trends_df[s]['LCOW']:.4f}")
        elif "process" not in s:
            if not hasattr(m.fs, "lcow_objective"):
                add_lcow_objective(m)
                print(f"Added LCOW objective for strategy: {s}")

            unfix_variables(m, s)
            set_optimization_bounds(m)
            solve_for_recovery(m, recovery=0.85, tee=False, display=True)

            micro_trends_df[s] = collect_micro_trend(m)
            macro_trends_df[s] = collect_macro_variables(m)
            total_mem_cost = sum(stage.costing.capital_cost.value for stage in m.fs.ro.values())
            levelized_mem_cost = total_mem_cost * m.fs.costing.total_investment_factor.value * m.fs.costing.capital_recovery_factor.value / m.fs.costing.annual_water_production()
            levelized_pump_cost = m.fs.pump.costing.capital_cost.value * m.fs.costing.total_investment_factor.value * m.fs.costing.capital_recovery_factor.value / m.fs.costing.annual_water_production()
            levelized_operating_cost = m.fs.costing.total_operating_cost.value / m.fs.costing.annual_water_production()
            macro_trends_df[s].update({
                "levelized_mem_cost": levelized_mem_cost,
                "levelized_pump_cost": levelized_pump_cost,
                "levelized_operating_cost": levelized_operating_cost
            })
            print(f"Completed strategy: {s} with LCOW = {macro_trends_df[s]['LCOW']:.4f}")
            print(f"Sh improvement factors: {[stage.feed_side.Sh_improvement_factor.value for stage in m.fs.ro.values()]}")
            print(f"Friction factor improvements: {[stage.feed_side.friction_factor_improvement.value for stage in m.fs.ro.values()]}")

        else:  # Process optimization involved
            if not hasattr(m.fs, "lcow_objective"):
                add_lcow_objective(m)
            unfix_variables(m, s)
            set_optimization_bounds(m)
            solve_for_recovery(m, recovery=0.85, tee=False, display=True)

            # Collect optimal l and pv for each stage
            l_opts = {i: m.fs.ro[i].length.value for i in m.fs.ro}
            pv_opts = {i: m.fs.ro[i].n_pressure_vessels.value for i in m.fs.ro}

            # Build possible values for each stage (floor and ceil)
            possible_l = {i: [np.floor(l_opts[i]), np.ceil(l_opts[i])] for i in m.fs.ro}
            possible_pv = {i: [np.floor(pv_opts[i]), np.ceil(pv_opts[i])] for i in m.fs.ro}

            from itertools import product

            # Prepare all combinations for all stages
            stage_ids = list(m.fs.ro.keys())
            grid_combinations = list(product(*[
                product(possible_l[i], possible_pv[i]) for i in stage_ids
            ]))

            process_grid_macros = []
            process_grid_micro_df = {}

            for combo in grid_combinations:
                # combo is a tuple of (l, pv) for each stage, in order of stage_ids
                for idx, (l, pv) in enumerate(combo):
                    stage = m.fs.ro[stage_ids[idx]]
                    stage.length.fix(l)
                    stage.n_pressure_vessels.fix(pv)
                try:
                    solve_for_recovery(m, recovery=0.85, tee=False, display=True)
                    macro_vars = collect_macro_variables(m)
                    total_mem_cost = sum(stage.costing.capital_cost.value for stage in m.fs.ro.values())
                    levelized_mem_cost = total_mem_cost * m.fs.costing.total_investment_factor.value * m.fs.costing.capital_recovery_factor.value / m.fs.costing.annual_water_production()
                    levelized_pump_cost = m.fs.pump.costing.capital_cost.value * m.fs.costing.total_investment_factor.value * m.fs.costing.capital_recovery_factor.value / m.fs.costing.annual_water_production()
                    levelized_operating_cost = m.fs.costing.total_operating_cost.value / m.fs.costing.annual_water_production()
                    macro_vars.update({
                        "levelized_mem_cost": levelized_mem_cost,
                        "levelized_pump_cost": levelized_pump_cost,
                        "levelized_operating_cost": levelized_operating_cost
                    })
                    # Store l and pv for all stages
                    for idx, (l, pv) in enumerate(combo):
                        macro_vars[f"l_stage{stage_ids[idx]}"] = l
                        macro_vars[f"pv_stage{stage_ids[idx]}"] = pv
                    process_grid_macros.append(macro_vars)
                    combo_key = ", ".join([f"{l},{pv}" for (l, pv) in combo])
                    process_grid_micro_df[combo_key] = collect_micro_trend(m)
                except Exception as e:
                    combo_key = ", ".join([f"{l},{pv}" for (l, pv) in combo])
                    print(f"Skipping infeasible case {combo_key}: {e}")

            process_grid_df = pd.DataFrame(process_grid_macros)
            best = min(process_grid_macros, key=lambda x: x["LCOW"])
            macro_trends_df[s] = {k: v for k, v in best.items() if
                                  not k.startswith("l_stage") and not k.startswith("pv_stage")}
            # Find the key for the best combination
            best_combo_key = ", ".join(
                [f"{best[f'l_stage{stage_id}']},{best[f'pv_stage{stage_id}']}" for stage_id in stage_ids])
            micro_trends_df[s] = process_grid_micro_df[best_combo_key]
            print(f"Completed strategy: {s} with optimal LCOW = {macro_trends_df[s]['LCOW']:.4f}")
            process_grid_df.to_excel(f"process_grid_results_bwro_{s}.xlsx", index=False)



    # Save the results to Excel
    macro_trends_df = pd.DataFrame(macro_trends_df).T
    print("\nMacro trends summary:")
    print(macro_trends_df.to_markdown(index=True))
    macro_trends_df["LCOW"] = pd.to_numeric(macro_trends_df["LCOW"], errors="coerce")
    opt_idx = macro_trends_df["LCOW"].idxmin()
    print(f"\nMost optimal design: {opt_idx} with LCOW = {macro_trends_df.loc[opt_idx, 'LCOW']:.4f}")
    with pd.ExcelWriter("ro_optimization_strategies_results_bwro.xlsx") as writer:
        for s, df in micro_trends_df.items():
            df.to_excel(writer, sheet_name=f"micro_{s}", index=False)
        macro_trends_df.to_excel(writer, sheet_name="macro_trends", index=True)
    return


def unfix_variables(m, strategy):
    m.fs.water_recovery.fix(0.85)
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


    print("Finished unfixing variables for strategy:", strategy)
    print(f"Degrees of freedom after using strategy: {degrees_of_freedom(m)}")
    return


def init_build_bwro_flowsheet(nfe=60, correlation_type="schock"):
    m = build_bwro_flowsheet(correlation_type=correlation_type, nfe=nfe)
    # Add telescoping potential variable and constraint to each stage
    for stage in m.fs.ro.values():
        add_telescoping_potential(stage)

    # Touch the osmotic pressure variable to ensure it is calculated
    for stage in m.fs.ro.values():
        for x in stage.feed_side.length_domain:
            stage.feed_side.properties[0, x].pressure_osm_phase[...]

    fix_model(m, ro_system="BWRO", velocity=0.25, salinity=5)
    scale_model(m, ro_system="BWRO")
    print(f"Degrees of freedom after fixing: {degrees_of_freedom(m)}")
    initialize_model(m, overpressure=6, ro_system="BWRO")
    print(f" Degrees of freedom after initialization: {degrees_of_freedom(m)}")
    solve(m, tee=False, display=False)
    print(f" Degrees of freedom after solving: {degrees_of_freedom(m)}")
    add_costing(m)
    print(f" Degrees of freedom after adding costing: {degrees_of_freedom(m)}")
    solve(m, tee=False, display=True)
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
    for s_id, stage in m.fs.ro.items():
        # Channel height between 0.1 mm and 2 mm
        stage.feed_side.channel_height.setlb(3e-4)
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
        stage.telescoping_potential.setlb(None)
        stage.telescoping_potential.setub(1.5)
        if s_id == 1:
            # Inlet velocity between 15 cm/s and 30 cm/s for optimal membrane performance, limits pressure vessel choice
            stage.feed_side.velocity[0, 0].setlb(0.15)
            stage.feed_side.velocity[0, 0].setub(0.30)
        # Exit velocity between 10 cm/s and 30 cm/s for optimal membrane performance, limits pressure vessel choice
        # stage.feed_side.velocity[0, 1].setlb(0.10)
        # Membrane length between 1 m and 10 m to represent the elements in a pressure vessel
        stage.length.setlb(1.0)
        stage.length.setub(10.0)
        # Remove the bound on the width of the membrane to allow for optimization
        stage.feed_side.width.setub(None)

        for x in stage.length_domain:
            stage.feed_side.N_Sh_comp[0, x, "TDS"].setub(1e3)
            stage.feed_side.friction_factor_darcy[0, x].setlb(1e-3)
            stage.feed_side.friction_factor_darcy[0, x].setub(100)



def collect_micro_trend(m):
    global_length_domain = []
    cumulative_length = 0.0
    # Build the global length domain
    for i in m.fs.ro_stages:
        stage = m.fs.ro[i]
        ld = list(stage.feed_side.length_domain)
        if len(ld) > 1:
            scaled_domain = [cumulative_length + x * value(stage.length) for x in
                             (ld if i == 1 else ld[1:])]
            global_length_domain.extend(scaled_domain)
            cumulative_length += value(stage.length)
    # Initialize dictionary for each key
    data = {'global_length_domain': global_length_domain}

    trend_dict = {}
    dens_solvent = m.fs.ro[1].dens_solvent.value
    for i in m.fs.ro_stages:
        stage = m.fs.ro[i]
        ld = list(stage.feed_side.length_domain)
        for x in ld if i ==1 else ld[1:]:
            trend_dict.setdefault('j_w', []).append(
                value(stage.flux_mass_phase_comp[0, x, "Liq", "H2O"]) / dens_solvent if x != 0 else 0)
            trend_dict.setdefault('k_f', []).append(value(stage.feed_side.K[0, x, "TDS"]))
            trend_dict.setdefault('cp', []).append(value(stage.feed_side.cp_modulus[0, x, 'TDS']))
            trend_dict.setdefault('bulk_osmotic_pressure_bar', []).append(
                value(stage.feed_side.properties[0, x].pressure_osm_phase['Liq']) / 1e5)
            trend_dict.setdefault('dp_dx_bar_per_m', []).append(
                value(stage.feed_side.dP_dx[0, x]) / 1e5)
            trend_dict.setdefault('Operating_pressure_bar', []).append(
                value(stage.feed_side.properties[0, x].pressure) / 1e5)
            trend_dict.setdefault('velocity', []).append(value(stage.feed_side.velocity[0, x]))
            trend_dict.setdefault('f', []).append(value(stage.feed_side.friction_factor_darcy[0, x]))

    data.update(trend_dict)
    data['cp_penalty'] = [(cp - 1) * osm_p for cp, osm_p in zip(data['cp'], data['bulk_osmotic_pressure_bar'])]
    dl = np.diff([0] + data['global_length_domain'])
    dp_dx = data['dp_dx_bar_per_m']
    dp = [dx * dpdx for dx, dpdx in zip(dl, dp_dx)]
    total_dP = -1 * np.cumsum([0] + dp)
    data['dp_friction'] = total_dP[1:]  # Remove the initial zero
    data['jw_kf'] = [jw / kf if kf != 0 else 0 for jw, kf in zip(data['j_w'], data['k_f'])]
    data['kinetic_energy'] = [0.5 * dens_solvent * v**2 for v in data['velocity']]
    # Create DataFrame
    df = pd.DataFrame(data)
    return df


    total_mem_cost = sum(stage.costing.capital_cost.value for stage in m.fs.ro.values())
    levelized_mem_cost = total_mem_cost * m.fs.costing.total_investment_factor.value * m.fs.costing.capital_recovery_factor.value / m.fs.costing.annual_water_production()
    levelized_pump_cost = m.fs.pump.costing.capital_cost.value * m.fs.costing.total_investment_factor.value * m.fs.costing.capital_recovery_factor.value / m.fs.costing.annual_water_production()
    levelized_operating_cost = m.fs.costing.total_operating_cost.value / m.fs.costing.annual_water_production()
    macro_trends_df[s].update({
        "levelized_mem_cost": levelized_mem_cost,
        "levelized_pump_cost": levelized_pump_cost,
        "levelized_operating_cost": levelized_operating_cost
    })


def solve(m, tee=False, display=False):
    solver = get_solver()
    sol = solver.solve(m, tee=tee)
    assert_optimal_termination(sol)
    if display:
        print_solved_state(m)
    return sol


def solve_for_recovery(m, recovery=0.5, tee=False, display=False):
    m.fs.water_recovery.fix(recovery)
    m.fs.pump.outlet.pressure[0].unfix()
    print(f"Solving for recovery of {recovery*100:.1f}% with DOF={degrees_of_freedom(m)}")
    sol = solve(m, tee=tee, display=display)
    return sol



# # Check the implementation
#     m = init_build_swro_flowsheet(nfe=5, correlation_type="schock")
#     # add_lcow_objective(m)
#     # unfix_variables(m, "process")
#     # set_optimization_bounds(m)
#     # solve_for_recovery(m, recovery=0.5, tee=False, display=True)
#     print(f"Optimal membrane length: {m.fs.ro[1].length.value:.2f} m")
#     print(f"Optimal number of pressure vessels: {m.fs.ro[1].n_pressure_vessels.value:.0f}")
#     print(f"Operating pressure: {m.fs.pump.outlet.pressure[0].value/1e5:.2f} bar")
#     m.fs.costing.pprint()
#     annual_operating_cost_per_m3 = m.fs.costing.total_operating_cost.value / m.fs.costing.annual_water_production()
#     print(f"Annual operating cost per m3: {annual_operating_cost_per_m3:.4f} USD/m3")
#     print(f"Optimal LCOW: {m.fs.costing.LCOW.expr():.4f} USD/m3")
#     mem_cost = m.fs.ro[1].costing.capital_cost.value * m.fs.costing.total_investment_factor.value * m.fs.costing.capital_recovery_factor.value / m.fs.costing.annual_water_production()
#     pump_cost = m.fs.pump.costing.capital_cost.value * m.fs.costing.total_investment_factor.value * m.fs.costing.capital_recovery_factor.value / m.fs.costing.annual_water_production()
#     print(f"Membrane cost: {mem_cost:.2f} USD/m3")
#     print(f"Pump cost: {pump_cost:.2f} USD/m3")
#     print(f"Ratio of membrane to pump cost: {mem_cost/pump_cost:.2f}")



if __name__ == "__main__":
    all_strategies = ["simulation", "topology", "module", "process", "topology_module_process"]
    # results = run_multiple_strategies(all_strategies, nfe=10, correlation_type="schock")
    for i in range(2, len(all_strategies) + 1):
        strategies_to_plot = all_strategies[:i]
        plot_multiple_strategies(strategies_to_plot=strategies_to_plot)



