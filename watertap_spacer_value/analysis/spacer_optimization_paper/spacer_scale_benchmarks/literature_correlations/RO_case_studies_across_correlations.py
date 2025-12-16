import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mticker
from watertap.core.solvers import get_solver
from watertap_spacer_value.flowsheets.ro_flowsheet_utils import *
import pandas as pd
from pyomo.opt import assert_optimal_termination
from idaes.core.util.model_statistics import degrees_of_freedom
from watertap_spacer_value.plotting_tools.paul_tol_color_maps import gen_paultol_colormap




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
        if ro_system == "SWRO":
            m.fs.pump.outlet.pressure[0].setub(85e5)
            for stage in m.fs.ro.values():
                for x in stage.feed_side.length_domain:
                    stage.feed_side.cp_modulus[0, x, "TDS"].setub(2.0)
        elif ro_system == "SWRO":
            m.fs.pump.outlet.pressure[0].setub(45e5)
            for stage in m.fs.ro.values():
                for x in stage.feed_side.length_domain:
                    stage.feed_side.cp_modulus[0, x, "TDS"].setub(2)
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





if __name__ == "__main__":
    run_case_studies(nfe=60, velocity=0.20, swro_salinity=35, bwro_salinity=5)





