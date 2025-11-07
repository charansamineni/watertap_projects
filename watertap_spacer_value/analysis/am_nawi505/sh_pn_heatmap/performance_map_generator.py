from idaes.core.util.initialization import propagate_state
from pyomo.environ import Var
from idaes.core.util import scaling as iscale
import numpy as np
import pandas as pd
from watertap_spacer_value.flowsheets.ro_flowsheet_utils import (
    build_bwro_flowsheet,
    build_swro_flowsheet,
    scale_model,
    add_costing,
)
from watertap_spacer_value.flowsheets.aqua_membrane_flowsheet_utils import (
    fix_aqua_membrane_case,
)
from watertap.core.solvers import get_solver
from pyomo.opt import assert_optimal_termination


def generate_data(n_samples, nfe=2, ro_type="SWRO", save_results=False):
    pn_range = np.logspace(6, 7, n_samples)
    sh_range = np.logspace(1, np.log10(100), n_samples)
    results_array = []


    for sh in sh_range:
        for pn in pn_range:

            if ro_type == "SWRO":
                m = build_swro_flowsheet(nfe=nfe, correlation_type="parameterized")
                salinity = 35
                recovery_target = 0.5
                overpressure = 3
            elif ro_type == "BWRO":
                m = build_bwro_flowsheet(nfe=nfe, correlation_type="parameterized")
                salinity = 5
                recovery_target = 0.85
                overpressure = 5.5 if sh < 50 else 5
            else:
                raise ValueError("Unknown ro_type")
            fix_aqua_membrane_case(
                m,
                velocity=0.25,
                salinity=salinity,
                channel_height=0.5e-3,
                porosity=0.99,
                n_vessels=180,
                ro_system=ro_type,
            )

            for stage in m.fs.ro.values():
                add_telescoping_potential(stage)
                stage.feed_side.sherwood_number[0].fix(sh)
                stage.feed_side.power_number[0].fix(pn)
                stage.telescoping_potential.setub(1.5)

            scale_model(m, ro_system=ro_type)
            scale_and_relax_bounds(m)
            initialize_model(m, ro_system=ro_type, overpressure=overpressure)
            print(f"We are after initialization for Sh: {sh}, Pn: {pn}")
            solve(m, tee=False)
            add_costing(m)
            solve(m, tee=False)
            print(f"We are after costing for Sh: {sh}, Pn: {pn}")
            # Fix recovery to target and unfix outlet pressure to simulate real operation
            m.fs.water_recovery.fix(recovery_target)
            m.fs.pump.outlet.pressure[0].unfix()
            try:
                print(f"We are after pumping for Sh: {sh}, Pn: {pn}")
                solve_res = solve(m, tee=False)
                if (
                    hasattr(solve_res, "solver")
                    and solve_res.solver.termination_condition.name != "optimal"
                ):
                    raise RuntimeError("Solver did not converge to optimal solution")
                telescoping_potential_val = sum(stage.telescoping_potential.value for stage in m.fs.ro.values()) / len(m.fs.ro)
                print(f"Telescoping potential: {telescoping_potential_val:.2f}")
                outputs_with_tags = {
                    "Sherwood number": sh,
                    "Power number": pn,
                    "LCOW": m.fs.costing.LCOW(),
                    "SEC": m.fs.costing.specific_energy_consumption(),
                    "Recovery": m.fs.water_recovery.value,
                    "Operating Pressure": m.fs.pump.outlet.pressure[0].value / 1e5,
                    "Telescoping Potential": telescoping_potential_val,
                    "CP modulus": sum(stage.feed_side.CP_avg[0, "TDS"]() for stage in m.fs.ro.values()) / len(m.fs.ro),
                }
                results_array.append(outputs_with_tags)
            except Exception as e:
                print(f"Solver failed for Sh: {sh}, Pn: {pn},-- {e}")
                outputs_with_tags = {
                    "Sherwood number": sh,
                    "Power number": pn,
                    "LCOW": np.nan,
                    "SEC": np.nan,
                    "Recovery": np.nan,
                    "Operating Pressure": np.nan,
                    "Telescoping Potential": np.nan,
                    "CP modulus": np.nan,
                }
                results_array.append(outputs_with_tags)

    if save_results:
        # Save results to Excel
        df = pd.DataFrame(results_array)
        df.to_excel(f"performance_map_{ro_type}_{n_samples}x{n_samples}_nfe{nfe}.xlsx", index=False)

    else:
        print(results_array)

    return



def scale_and_relax_bounds(m):

    m.fs.pump.control_volume.properties_out[0].pressure.setub(5e8)
    m.fs.pump.control_volume.properties_out[0].pressure.setlb(1e5)
    m.fs.pump.outlet.pressure[0].setub(5e8)

    for stage in m.fs.ro.values():
        for p in stage.feed_side.properties_interface.values():
            p.pressure.setub(5e8)

        if hasattr(stage.feed_side, "properties_in"):
            for p in stage.feed_side.properties_in.values():
                p.pressure.setub(5e8)
            for p in stage.feed_side.properties_out.values():
                p.pressure.setub(5e8)
        else:
            for p in stage.feed_side.properties.values():
                p.pressure.setub(5e8)

    for stage in m.fs.ro.values():
        stage.feed_side.sherwood_number[0].setlb(1)
        stage.feed_side.sherwood_number[0].setub(1000)
        stage.feed_side.power_number[0].setlb(1e4)
        stage.feed_side.power_number[0].setub(5e7)

        iscale.set_variable_scaling_from_current_value(
            stage.feed_side.sherwood_number[0]
        )
        iscale.set_variable_scaling_from_current_value(stage.feed_side.power_number[0])

        for v in stage.feed_side.velocity.values():
            v.setlb(1e-4)
            v.setub(1e2)

        for v in stage.feed_side.N_Re.values():
            v.setlb(1e-3)
            v.setub(1e3)

        for v in stage.feed_side.friction_factor_darcy.values():
            v.setlb(1e-3)
            v.setub(1e2)

        for v in stage.feed_side.N_Sh_comp.values():
            v.setlb(1)
            v.setub(1000)

        for v in stage.feed_side.K.values():
            v.setlb(1e-8)
            v.setub(1e-1)

        for v in stage.feed_side.cp_modulus.values():
            v.setlb(1)
            v.setub(100)

        for v in stage.deltaP.values():
            v.setlb(-1e8)
            v.setub(0)

        for v in stage.feed_side.dP_dx.values():
            v.setlb(-1e8)
            v.setub(None)

        if hasattr(stage.feed_side, "properties_in"):
            for j in stage.feed_side.config.property_package.component_list:
                stage.feed_side.properties_in[0].flow_mass_phase_comp["Liq", j].setlb(
                    1e-12
                )
                stage.feed_side.properties_in[0].flow_mass_phase_comp["Liq", j].setub(
                    1e4
                )
        else:
            for j in stage.feed_side.config.property_package.component_list:
                for x in stage.feed_side.length_domain:
                    stage.feed_side.properties[0, x].flow_mass_phase_comp[
                        "Liq", j
                    ].setlb(1e-12)
                    stage.feed_side.properties[0, x].flow_mass_phase_comp[
                        "Liq", j
                    ].setub(1e4)


def add_telescoping_potential(stage):
    stage.telescoping_potential = Var(
        initialize=1.0, bounds=(1e-3, 1.5), doc="Telescoping potential variable",
    )

    @stage.Constraint(doc="Telescoping potential constraint",)
    def telescoping_potential_constraint(b):
        return (
            b.telescoping_potential * b.length * 1e5 == -1 * b.deltaP[0]
        )  # Convert Pa to bar


def solve(m, tee=False):
    solver = get_solver()
    results = solver.solve(m, tee=tee)
    assert_optimal_termination(results)
    return results


def initialize_model(m, overpressure=2, ro_system="SWRO", verbose=False):
    # Touch the osmotic pressure variables to ensure they are initialized
    m.fs.feed.properties[0].pressure_osm_phase[...]
    m.fs.feed.properties[0].flow_vol_phase[...]
    m.fs.feed.properties[0].conc_mass_phase_comp[...]
    m.fs.feed.initialize()

    propagate_state(m.fs.feed_to_pump)
    osm_pressure = m.fs.feed.properties[0].pressure_osm_phase["Liq"].value
    print(f"Osmotic pressure of feed: {osm_pressure * 1e-5:.2f} bar")
    m.fs.pump.outlet.pressure[0].fix(overpressure * osm_pressure)
    m.fs.pump.initialize()
    if verbose:
        m.fs.pump.report()

    # Initialize RO stages
    if ro_system == "SWRO":
        propagate_state(m.fs.pump_to_ro)
        m.fs.ro[1].initialize()
        if verbose:
            m.fs.ro[1].report()
    elif ro_system == "BWRO":
        # Initialize each RO stage sequentially
        for s_id, stage in m.fs.ro.items():
            if hasattr(m.fs, "ro_inlet_arc_" + str(s_id)):
                if verbose:
                    # print the arc
                    print(f"Initializing RO stage {s_id} with inlet arc")
                    getattr(m.fs, "ro_inlet_arc_" + str(s_id)).pprint()

            propagate_state(getattr(m.fs, "ro_inlet_arc_" + str(s_id)))
            stage.initialize()
            if verbose:
                stage.report()
            # propagate the permeate stream to the mixer
            propagate_state(getattr(m.fs, f"stage_{s_id}_to_P_mixer"))

        # Initialize the permeate mixer after all RO stages are initialized
        m.fs.P_mixer.initialize()

    # Initialize product and brine streams
    m.fs.product.properties[0].flow_vol_phase[...]
    m.fs.product.properties[0].conc_mass_phase_comp[...]
    if ro_system == "SWRO":
        propagate_state(m.fs.ro_to_product)
    elif ro_system == "BWRO":
        propagate_state(m.fs.P_mixer_to_product)
    m.fs.product.initialize()

    propagate_state(m.fs.retentate_to_erd)
    m.fs.erd.initialize()

    m.fs.brine.properties[0].flow_vol_phase[...]
    m.fs.brine.properties[0].conc_mass_phase_comp[...]
    propagate_state(m.fs.erd_to_brine)
    m.fs.brine.initialize()

    if verbose:
        m.fs.product.report()
        m.fs.erd.report()
        m.fs.brine.report()

if __name__ == "__main__":
    n_samples =2
    nfe =2
    for ro in ["BWRO"]:
        print(f"Generating data for {ro} with {n_samples} samples and {nfe} finite elements")
        generate_data(n_samples=n_samples, nfe=nfe, ro_type=ro, save_results=False)
