from watertap_spacer_value.analysis.spacer_optimization_paper.correlation_comparison.RO_case_studies_across_correlations import (
    solve,
    solve_for_recovery,
)
from watertap_spacer_value.flowsheets.ro_flowsheet_utils import *
from math import ceil, floor
import pandas as pd
from pyomo.environ import Var, Objective
from idaes.core.util.model_statistics import degrees_of_freedom


def main():

    results = {}
    m = build_swro_flowsheet(correlation_type="guillen", nfe=10)
    # Ensure osmotic pressures are initialized on the feed side
    for stage in m.fs.ro.values():
        for x in stage.feed_side.length_domain:
            stage.feed_side.properties[0, x].pressure_osm_phase[...]

    fix_model(m, velocity=0.25, salinity=35, ro_system="SWRO")
    scale_model(m, ro_system="SWRO")
    assert degrees_of_freedom(m) == 1, "Model is not fully specified after fixing."
    initialize_model(m, overpressure=2, ro_system="SWRO")
    assert degrees_of_freedom(m) == 0, "DOF is not zero after initialization."
    solve(m, tee=False, display=False)
    add_costing(m)
    solve(m, tee=False, display=False)
    solve_for_recovery(m, recovery=0.5, tee=False, display=True, strategy="simulation")
    macro_vars = collect_macro_variables(m)
    results["base_case"] = macro_vars
    l, n_pv = cont_optimize_process(m, tee=False)
    print(f"Optimal membrane length: {l:.2f} m")
    print(f"Optimal number of pressure vessels: {n_pv:.0f}")
    possible_pvs = [floor(n_pv), ceil(n_pv)]
    possible_l = [floor(l), ceil(l)]
    for pl in possible_l:
        for n in possible_pvs:
            print(f"Trying membrane length of {pl} m with {n} pressure vessels.")
            for stage in m.fs.ro.values():
                stage.length.fix(pl)
                stage.n_pressure_vessels.fix(n)

            solve(m, tee=False, display=False)
            macro_vars = collect_macro_variables(m)
            results[f"opt_l{pl}_npv{n}"] = macro_vars

    df = pd.DataFrame(results).T
    print(df.to_markdown(index=True))
    df["LCOW"] = pd.to_numeric(df["LCOW"], errors="coerce")
    opt_idx = df["LCOW"].idxmin()
    print(f"Most optimal design: {opt_idx} with LCOW = {df.loc[opt_idx, 'LCOW']:.4f}")
    return


def cont_optimize_process(m, tee=False):
    for stage in m.fs.ro.values():
        stage.telescoping_potential = Var(
            initialize=1.0, bounds=(0.1, 1.5), doc="Telescoping potential",
        )

        @stage.Constraint(doc="Telescoping potential definition")
        def telescoping_potential_defn(b):
            return b.telescoping_potential * b.length * -1e5 == b.deltaP[0]

    m.fs.lcow_objective = Objective(expr=m.fs.costing.LCOW)

    m.fs.water_recovery.fix(0.5)
    m.fs.pump.outlet.pressure[0].unfix()

    for stage in m.fs.ro.values():
        stage.n_pressure_vessels.unfix()
        stage.length.unfix()
        stage.feed_side.channel_height.unfix()
        stage.feed_side.spacer_porosity.unfix()
        stage.feed_side.Sh_improvement_factor.unfix()
        stage.feed_side.friction_factor_improvement.unfix()

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
        # Inlet velocity between 15 cm/s and 30 cm/s for optimal membrane performance, limits pressure vessel choice
        stage.feed_side.velocity[0, 0].setlb(0.15)
        stage.feed_side.velocity[0, 0].setub(0.30)
        # Membrane length between 1 m and 10 m to represent the elements in a pressure vessel
        stage.length.setlb(1.0)
        stage.length.setub(10.0)
        # Remove the bound on the width of the membrane to allow for optimization
        stage.feed_side.width.setub(None)

    print(f" Degrees of freedom after unfixing: {degrees_of_freedom(m)}")
    solve(m, tee=tee, display=True)
    opt_length = m.fs.ro[1].length.value
    opt_npv = m.fs.ro[1].n_pressure_vessels.value
    return opt_length, opt_npv


if __name__ == "__main__":
    main()
