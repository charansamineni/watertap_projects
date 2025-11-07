from pyomo.core import Objective, Var
from watertap_spacer_value.flowsheets.bwro_321 import build_bwro_flowsheet as build_model
from watertap_spacer_value.analysis.correlation_comparison.bwro_321 import fix_model, scale_model, initialize_model
from watertap_spacer_value.flowsheets.ro_flowsheet_utils import (
    add_costing,
    degrees_of_freedom,
    solve,
)
from math import ceil, floor


def main():
    bwro = init_build_swro_flowsheet(nfe=60)
    cont_optimize_process(bwro, tee=True)
    return bwro



def init_build_swro_flowsheet(nfe=60):
    m = build_model(correlation_type="guillen", nfe=nfe)
    fix_model(m)
    scale_model(m)
    print(f"Degrees of freedom after fixing: {degrees_of_freedom(m)}")
    initialize_model(m, overpressure=6.25)
    print(f" Degrees of freedom after initialization: {degrees_of_freedom(m)}")
    solve(m, tee=False, display=False)
    print(f" Degrees of freedom after solving: {degrees_of_freedom(m)}")
    add_costing(m)
    print(f" Degrees of freedom after adding costing: {degrees_of_freedom(m)}")
    solve(m, tee=False, display=True)
    print(f" Degrees of freedom after solving with costing: {degrees_of_freedom(m)}")
    return m


def cont_optimize_process(m, tee=False):
    for stage in m.fs.ro.values():
        stage.telescoping_potential = Var(
            initialize=1.0,
            bounds=(0.1, 1.5),
            doc="Telescoping potential",
        )

        @stage.Constraint(doc="Telescoping potential definition")
        def telescoping_potential_defn(b):
            return b.telescoping_potential * b.length * -1e5 == b.deltaP[0]

    m.fs.lcow_objective = Objective(expr=m.fs.costing.LCOW)

    m.fs.water_recovery.fix(0.85)
    m.fs.pump.outlet.pressure[0].unfix()

    for s_id, stage in m.fs.ro.items():
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
        # Exit velocity between 10 cm/s and 30 cm/s for optimal membrane performance, limits pressure vessel choice
        stage.feed_side.velocity[0, 1].setlb(0.10)
        # Membrane length between 1 m and 10 m to represent the elements in a pressure vessel
        stage.length.setlb(1.0)
        stage.length.setub(10.0)
        # Remove the bound on the width of the membrane to allow for optimization
        stage.feed_side.width.setub(None)

    solve(m, tee=tee, display=True)

    print(f" Degrees of freedom after unfixing: {degrees_of_freedom(m)}")
    for i, stage in m.fs.ro.items():
        opt_length = stage.length.value
        opt_npv = stage.n_pressure_vessels.value
        op_sh = stage.feed_side.Sh_improvement_factor.value
        op_fric = stage.feed_side.friction_factor_improvement.value
        opt_h = stage.feed_side.channel_height.value
        opt_por = stage.feed_side.spacer_porosity.value
        print(
            f"Stage {i}:\n"
            f"  Optimal channel height: {opt_h * 1e3:.2f} mm\n"
            f"  Optimal spacer porosity: {opt_por:.3f} (-)\n"
            f"  Optimal membrane length: {opt_length:.2f} m\n"
            f"  Optimal number of pressure vessels: {opt_npv:.0f}\n"
            f"  Optimal Sherwood number improvement factor: {op_sh:.2f}\n"
            f"  Optimal friction factor improvement factor: {op_fric:.2f}"
        )

    return



if __name__ == "__main__":
    main()
