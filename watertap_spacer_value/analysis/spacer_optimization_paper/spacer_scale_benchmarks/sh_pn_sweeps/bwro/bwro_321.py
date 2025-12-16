import numpy as np
import pandas as pd
from pyomo.environ import Var
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util import scaling as iscale
from watertap_spacer_value.flowsheets.ro_flowsheet_utils import (
    build_bwro_flowsheet as build_model,
    fix_model, initialize_model, scale_model
)
from watertap_spacer_value.flowsheets.ro_flowsheet_utils import (
    add_costing,
)
from watertap_spacer_value.analysis.spacer_optimization_paper.spacer_scale_benchmarks.literature_correlations.RO_case_studies_across_correlations import solve


def sh_pn_sweep_bwro(nfe=2):

    pn_range = np.logspace(np.log10(1e5), np.log10(1e7), 15)
    sh_range = np.logspace(1, np.log10(200), 15)

    results_array = []
    list_results_array = []

    for sh in sh_range:
        for pn in pn_range:
            try:
                print(f"Running for Sh: {sh}, Pn: {pn}")
                m = build_model(correlation_type="parameterized", nfe=nfe)
                fix_model(m, velocity=0.20, salinity=5, ro_system="BWRO")
                for stage in m.fs.ro.values():
                    add_telescoping_potential(stage)
                    stage.feed_side.sherwood_number[0].fix(sh)
                    stage.feed_side.power_number[0].fix(pn)
                    stage.telescoping_potential.setub(1.5)

                scale_model(m, ro_system="BWRO")
                scale_and_relax_bounds(m)
                print(f"Degrees of freedom after fixing: {degrees_of_freedom(m)}")
                initialize_model(m, overpressure=6, ro_system="BWRO")
                print(f" Degrees of freedom after initialization: {degrees_of_freedom(m)}")
                solve(m, tee=False, display=False)
                print(f" Degrees of freedom after solving: {degrees_of_freedom(m)}")
                add_costing(m)
                print(f" Degrees of freedom after adding costing: {degrees_of_freedom(m)}")
                solve(m, tee=False, display=False)
                print(f" Degrees of freedom after solving with costing: {degrees_of_freedom(m)}")



                m.fs.water_recovery.fix(0.85)
                m.fs.pump.outlet.pressure[0].unfix()
                print(f"Degrees of freedom after unfixing: {degrees_of_freedom(m)}")
                print(f"Solving for Sh: {sh}, Pn: {pn}")
                solve(m, tee=False, display=True)
                print(f"Telescoping potential: {m.fs.ro[1].telescoping_potential.value:.2f}")

                # Collect outputs
                outputs, listed_outputs = build_outputs(m)
                result = {"Sherwood number": sh, "Power number": pn}
                result.update({k: float(v) if hasattr(v, "value") else v for k, v in outputs.items()})
                results_array.append(result)
                list_result = {"Sherwood number": sh, "Power number": pn}
                list_result.update({k: [float(i) for i in v] for k, v in listed_outputs.items()})
                list_results_array.append(list_result)
            except Exception as e:
                print(f"Failed for Sh: {sh}, Pn: {pn} with error: {e}")
                print(f"Solver failed for Sh: {sh}, Pn: {pn},-- {e}")
                result = {"Sherwood number": sh, "Power number": pn}
                # Use np.nan for all expected outputs
                # outputs, listed_outputs = build_outputs(m)
                result.update({k: np.nan for k in outputs.keys()})
                results_array.append(result)
                list_result = {"Sherwood number": sh, "Power number": pn}
                list_result.update({k: [np.nan for _ in v] for k, v in listed_outputs.items()})
                list_results_array.append(list_result)


    # Save results to Excel
    df = pd.DataFrame(results_array)
    df.to_csv("bwro_sweep_results.csv", index=False)

    df_list = pd.DataFrame(list_results_array)
    df_list.to_csv("bwro_sweep_stagewise_results.csv", index=False)

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
        stage.feed_side.sherwood_number[0].setub(600)
        stage.feed_side.power_number[0].setlb(5e4)
        stage.feed_side.power_number[0].setub(5e7)

        iscale.set_variable_scaling_from_current_value(
            stage.feed_side.sherwood_number[0]
        )
        iscale.set_variable_scaling_from_current_value(stage.feed_side.power_number[0])

        for v in stage.feed_side.velocity.values():
            v.setlb(1e-4)
            v.setub(1e2)

        for v in stage.feed_side.N_Sh_comp.values():
            v.setlb(1)
            v.setub(600)

        for v in stage.feed_side.N_Re.values():
            v.setlb(0.01)
            v.setub(1e3)

        for v in stage.feed_side.friction_factor_darcy.values():
            v.setlb(1e-3)
            v.setub(1e2)

        for v in stage.feed_side.K.values():
            v.setlb(1e-8)
            v.setub(1e-1)

        for v in stage.feed_side.cp_modulus.values():
            v.setlb(0.01)
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
        initialize=1.0,
        bounds=(1e-3, 1.5),
        doc="Telescoping potential variable",
    )
    @stage.Constraint(
        doc = "Telescoping potential constraint",
    )
    def telescoping_potential_constraint(b):
        return b.telescoping_potential * b.length * 1e5 == -1 * b.deltaP[0] # Convert Pa to bar



def build_outputs(m):
    outputs = {
        "LCOW": m.fs.costing.LCOW(),
        "Energy Consumption": m.fs.costing.specific_energy_consumption(),
        "Volumetric Recovery": m.fs.water_recovery.value,
    }

    # Calculate averages across all RO stages
    n_stages = len(m.fs.ro)
    area_avg = sum(stage.area.value for stage in m.fs.ro.values()) / n_stages
    area_list = [stage.area.value for stage in m.fs.ro.values()]
    cp_avg = sum(stage.feed_side.CP_avg[0, "TDS"]() for stage in m.fs.ro.values()) / n_stages
    cp_list = [stage.feed_side.CP_avg[0, "TDS"]() for stage in m.fs.ro.values()]
    deltaP_avg = sum(stage.deltaP[0].value for stage in m.fs.ro.values()) / n_stages
    deltaP_list = [stage.deltaP[0].value for stage in m.fs.ro.values()]
    K_avg = sum(stage.feed_side.K_avg[0, "TDS"]() for stage in m.fs.ro.values()) / n_stages
    K_list = [stage.feed_side.K_avg[0, "TDS"]() for stage in m.fs.ro.values()]
    f_avg = sum(stage.feed_side.f_avg[0]() for stage in m.fs.ro.values()) / n_stages
    Re_avg = sum(stage.feed_side.N_Re_avg[0]() for stage in m.fs.ro.values()) / n_stages
    Re_list = [stage.feed_side.N_Re_avg[0]() for stage in m.fs.ro.values()]
    Sh_avg = sum(stage.feed_side.Sh_avg[0, "TDS"]() for stage in m.fs.ro.values()) / n_stages
    Sh_list = [stage.feed_side.Sh_avg[0, "TDS"]() for stage in m.fs.ro.values()]
    Pn_avg = sum(stage.feed_side.Pn_avg[0]() for stage in m.fs.ro.values()) / n_stages
    Pn_list = [stage.feed_side.Pn_avg[0]() for stage in m.fs.ro.values()]
    telescoping_avg = sum(stage.telescoping_potential.value for stage in m.fs.ro.values()) / n_stages
    telescoping_list = [stage.telescoping_potential.value for stage in m.fs.ro.values()]
    inlet_v_list = [stage.feed_side.velocity[0, 0] for stage in m.fs.ro.values()]
    outlet_v_list = [stage.feed_side.velocity[0, 1] for stage in m.fs.ro.values()]
    pv_list = [stage.n_pressure_vessels.value for stage in m.fs.ro.values()]
    dp_total = sum(stage.deltaP[0].value for stage in m.fs.ro.values())

    outputs.update({
        "Average membrane area": area_avg,
        "Average cp (TDS)": cp_avg,
        "Average pressure drop": deltaP_avg,
        "Average K (TDS)": K_avg,
        "Average f": f_avg,
        "Average Re": Re_avg,
        "Average Sh (TDS)": Sh_avg,
        "Average Pn": Pn_avg,
        "Average telescoping potential": telescoping_avg,
        "Total pressure drop": dp_total,
    })

    # Add representative values from first and last stage
    outputs.update({
        "Inlet velocity": m.fs.ro[1].feed_side.velocity[0, 0].value,
        "Outlet velocity": m.fs.ro[n_stages].feed_side.velocity[0, 1].value,
        "Power number": m.fs.ro[1].feed_side.power_number[0].value,
        "Channel height": m.fs.ro[1].feed_side.channel_height.value,
        "Spacer porosity": m.fs.ro[1].feed_side.spacer_porosity.value,
        "Length": m.fs.ro[1].length.value,
        "Number of pressure vessels": m.fs.ro[1].n_pressure_vessels.value,
        "Operating pressure": m.fs.pump.outlet.pressure[0].value,
        "Outlet bulk concentration": m.fs.ro[n_stages].feed_side.properties[0, 1].conc_mass_phase_comp[
            "Liq", "TDS"].value,
        "Outlet interface concentration": m.fs.ro[n_stages].feed_side.properties_interface[0, 1].conc_mass_phase_comp[
            "Liq", "TDS"].value,
    })

    # Create a second outputs array for saving to a different sheet
    outputs_stagewise = {
        "Membrane area list": area_list,
        "cp (TDS) list": cp_list,
        "Pressure drop list": deltaP_list,
        "K (TDS) list": K_list,
        "f list": [stage.feed_side.f_avg[0]() for stage in m.fs.ro.values()],
        "Re list": Re_list,
        "Sh (TDS) list": Sh_list,
        "Pn list": Pn_list,
        "Telescoping potential list": telescoping_list,
        "Inlet velocity list": [v.value for v in inlet_v_list],
        "Outlet velocity list": [v.value for v in outlet_v_list],
        "Number of pressure vessels list": pv_list,
    }

    return outputs, outputs_stagewise




if __name__ == "__main__":
    sh_pn_sweep_bwro(nfe=10)
