import numpy as np
from pyomo.opt import assert_optimal_termination
from watertap.core.solvers import get_solver
from watertap_spacer_value.flowsheets.ro_flowsheet_utils import *
from pyomo.environ import Var, Objective
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util import scaling as iscale
from watertap_spacer_value.analysis.spacer_optimization_paper.spacer_scale_benchmarks.literature_correlations.RO_case_studies_across_correlations import (
    solve,
    solve_for_recovery,
)


def sweep_build(
    nfe=60, velocity=0.25, ro_system="SWRO", salinity=35, correlation_type="schock"
):
    if ro_system.upper() == "SWRO":
        m = build_swro_flowsheet(correlation_type=correlation_type, nfe=nfe)
    elif ro_system.upper() == "BWRO":
        m = build_bwro_flowsheet(correlation_type=correlation_type, nfe=nfe)
    else:
        raise ValueError("ro_system must be either 'SWRO' or 'BWRO'")

    # Ensure osmotic pressures are initialized on the feed side
    for stage in m.fs.ro.values():
        add_telescoping_potential(stage)
        for x in stage.feed_side.length_domain:
            stage.feed_side.properties[0, x].pressure_osm_phase[...]

    fix_model(m, velocity=velocity, salinity=salinity, ro_system=ro_system.upper())
    scale_model(m, ro_system=ro_system.upper())

    # Relax bounds for better convergence during initialization for BWRO systems
    if ro_system == "BWRO":
        set_low_salinity_bounds(m)

    # Assert degrees of freedom after fixing and initialize the model
    assert degrees_of_freedom(m) == 1, "Model is not fully specified after fixing."
    initialize_model(m, overpressure=2 if ro_system.upper() == "SWRO" else 7, ro_system=ro_system.upper())
    assert degrees_of_freedom(m) == 0, "DOF is not zero after initialization."

    # Solve the model at the initialized operating pressure
    solve(m, tee=False, display=False)

    # Add costing to the model and solve again
    add_costing(m)
    solve(m, tee=False, display=False)

    # # Add constraints to limit maximum operating pressure and concentration polarization modulus
    # if ro_system == "SWRO":
    #     m.fs.pump.outlet.pressure[0].setub(85e5)
    #     for stage in m.fs.ro.values():
    #         for x in stage.feed_side.length_domain:
    #             stage.feed_side.cp_modulus[0, x, "TDS"].setub(2.0)
    # elif ro_system == "SWRO":
    #     m.fs.pump.outlet.pressure[0].setub(45e5)
    #     for stage in m.fs.ro.values():
    #         for x in stage.feed_side.length_domain:
    #             stage.feed_side.cp_modulus[0, x, "TDS"].setub(2)

    # Solve for the desired recovery again after adding new constraints

    desired_recovery = 0.5 if ro_system == "SWRO" else 0.85
    print(f"desired_recovery: {desired_recovery}")
    solve_for_recovery(
        m, recovery=desired_recovery, tee=False, display=True, strategy='simulation'
    )

    return m


if __name__ == "__main__":
    model = sweep_build(
        nfe=10, velocity=0.20, ro_system="BWRO", salinity=5, correlation_type="schock",
    )
