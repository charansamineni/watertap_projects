from pyomo.environ import Var
from idaes.core.util import scaling as iscale
import numpy as np
import pandas as pd
from watertap_spacer_value.flowsheets.swro_single_stage import (
    build_swro_flowsheet as build_model,
)
from watertap_spacer_value.analysis.correlation_comparison.swro_single_stage import fix_model, scale_model, initialize_model
from watertap_spacer_value.flowsheets.ro_flowsheet_utils import (
    add_costing,
    degrees_of_freedom,
    solve,
)
from watertap_spacer_value.analysis.sh_pn_sweeps.bwro.bwro_321 import build_outputs


def sh_pn_sweep_swro(nfe=2, save_results=False):

    pn_range = np.logspace(5, 7, 15)
    sh_range = np.logspace(1, np.log10(200), 15)

    # pn_range = [1e6]
    # sh_range = [200]

    results_array = []
    list_results_array = []

    for sh in sh_range:
        for pn in pn_range:
            m = build_model(correlation_type="parameterized", nfe=nfe)
            fix_model(m)

            for stage in m.fs.ro.values():
                add_telescoping_potential(stage)
                stage.feed_side.sherwood_number[0].fix(sh)
                stage.feed_side.power_number[0].fix(pn)
                stage.telescoping_potential.setub(1.5)

            scale_model(m)
            scale_and_relax_bounds(m)
            print(f"Degrees of freedom after fixing: {degrees_of_freedom(m)}")
            initialize_model(m, overpressure=2)
            print(
                f" Degrees of freedom after initialization: {degrees_of_freedom(m)}")
            solve(m, tee=False, display=False)
            print(f" Degrees of freedom after solving: {degrees_of_freedom(m)}")
            add_costing(m)
            print(f" Degrees of freedom after adding costing: {degrees_of_freedom(m)}")
            solve(m, tee=False, display=False)
            print(f" Degrees of freedom after solving with costing: {degrees_of_freedom(m)}")


            m.fs.water_recovery.fix(0.5)
            m.fs.pump.outlet.pressure[0].unfix()
            print(f"Degrees of freedom after unfixing: {degrees_of_freedom(m)}")
            print(f"Solving for Sh: {sh}, Pn: {pn}")
            try:
                solve_res = solve(m, tee=False, display=True)
                if hasattr(solve_res, "solver") and solve_res.solver.termination_condition.name != "optimal":
                    raise RuntimeError("Solver did not converge to optimal solution")
                telescoping_potential_val = m.fs.ro[1].telescoping_potential.value
                print(f"Telescoping potential: {telescoping_potential_val:.2f}")
                outputs, listed_outputs = build_outputs(m)
                result = {"Sherwood number": sh, "Power number": pn}
                result.update({k: float(v) if hasattr(v, "value") else v for k, v in outputs.items()})
                results_array.append(result)
                list_result = {"Sherwood number": sh, "Power number": pn}
                list_result.update({k: [float(i) for i in v] for k, v in listed_outputs.items()})
                list_results_array.append(list_result)
            except Exception as e:
                print(f"Solver failed for Sh: {sh}, Pn: {pn},-- {e}")
                result = {"Sherwood number": sh, "Power number": pn}
                # Use np.nan for all expected outputs
                outputs, listed_outputs = build_outputs(m)
                result.update({k: np.nan for k in outputs.keys()})
                results_array.append(result)
                list_result = {"Sherwood number": sh, "Power number": pn}
                list_result.update({k: [np.nan for _ in v] for k, v in listed_outputs.items()})
                list_results_array.append(list_result)


    if save_results:
        # Save results to Excel
        df = pd.DataFrame(results_array)
        df.to_csv("swro_sweep_results.csv", index=False)

        df_list = pd.DataFrame(list_results_array)
        df_list.to_csv("swro_sweep_stagewise_results.csv", index=False)
    else:
        print(results_array)

    return

def sweep_build(nfe=2, velocity=0.25, salinity=35):
    m = build_model(correlation_type="parameterized", nfe=nfe)
    fix_model(m , velocity=velocity, salinity=salinity)
    scale_model(m)
    scale_and_relax_bounds(m)
    print(f"Degrees of freedom after fixing: {degrees_of_freedom(m)}")
    initialize_model(m, overpressure=2.5)
    print(f" Degrees of freedom after initialization: {degrees_of_freedom(m)}")
    solve(m, tee=False, display=False)
    add_costing(m)
    solve(m, tee=False, display=False)
    for stage in m.fs.ro.values():
        add_telescoping_potential(stage)
    # Fix recovery to 50% and unfix outlet pressure to simulate real operation
    m.fs.water_recovery.fix(0.5)
    m.fs.pump.outlet.pressure[0].unfix()
    solve(m, tee=True, display=False)
    print(f"Telescoping potential: {m.fs.ro[1].telescoping_potential.value:.2f}")
    return m


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
        initialize=1.0,
        bounds=(1e-3, 1.5),
        doc="Telescoping potential variable",
    )
    @stage.Constraint(
        doc = "Telescoping potential constraint",
    )
    def telescoping_potential_constraint(b):
        return b.telescoping_potential * b.length * 1e5 == -1 * b.deltaP[0] # Convert Pa to bar



# def build_outputs(m):
#     outputs = {
#         "LCOW": m.fs.costing.LCOW,
#         "Energy Consumption": m.fs.costing.specific_energy_consumption,
#         "Volumetric Recovery": m.fs.water_recovery,
#     }
#
#     # Collect outputs for all RO stage 1
#     stage = m.fs.ro[1]
#     outputs.update(
#         {
#             f"Membrane area": stage.area,
#             f"cp_average": stage.feed_side.CP_avg[0, "TDS"],
#             f"Pressure drop": stage.deltaP[0],
#             f"Average K": stage.feed_side.K_avg[0, "TDS"],
#             f"Average f": stage.feed_side.f_avg[0],
#             f"Average Re": stage.feed_side.N_Re_avg[0],
#             f"Average Sh": stage.feed_side.Sh_avg[0, "TDS"],
#             f"Average Pn": stage.feed_side.Pn_avg[0],
#             f"Inlet velocity": stage.feed_side.velocity[0, 0],
#             f"Outlet velocity": stage.feed_side.velocity[0, 1],
#             f"Sherwood number": stage.feed_side.sherwood_number[0],
#             f"Power number": stage.feed_side.power_number[0],
#             f"Channel height": stage.feed_side.channel_height,
#             f"Spacer porosity": stage.feed_side.spacer_porosity,
#             f"Length": stage.length,
#             f"Number of pressure vessels": stage.n_pressure_vessels,
#             f"Telescoping potential": stage.telescoping_potential,
#             f"Operating pressure": m.fs.pump.outlet.pressure[0],
#             f"Outlet bulk concentration": (stage.feed_side.properties[0, 1].conc_mass_phase_comp["Liq", "TDS"]
#             ),
#             f"Outlet interface concentration": stage.feed_side.properties_interface[
#                 0, 1
#             ].conc_mass_phase_comp["Liq", "TDS"],
#         }
#     )
#
#     return outputs


# def sh_pn_sweep_swro(n_samples=3):
#
#     opt_func = solve_for_recovery
#     opt_kwargs = {"recovery": 0.5, "tee": False, "display": False}
#
#     kwargs_dict = {
#         "h5_results_file_name": "sh_pn_sweep_swro.h5",
#         "build_model": sweep_build,
#         "build_model_kwargs": dict(nfe=2),
#         "build_sweep_params": build_sweep_params,
#         "build_sweep_params_kwargs": dict(n_samples=n_samples),
#         "build_outputs": build_outputs,
#         "build_outputs_kwargs": {},
#         "optimize_function": opt_func,
#         "optimize_kwargs": opt_kwargs,
#         "initialize_function": None,
#         "initialize_kwargs": {},
#         "parallel_back_end": "ConcurrentFutures",
#         "number_of_subprocesses": 1,
#         "csv_results_file_name": "sh_pn_sweep_swro.csv",
#         "h5_parent_group_name": None,
#         "update_sweep_params_before_init": False,
#         "initialize_before_sweep": False,
#         "reinitialize_function": None,
#         "reinitialize_kwargs": {},
#         "reinitialize_before_sweep": False,
#         "probe_function": None,
#         "interpolate_nan_outputs": True,
#     }
#
#     # create parameter sweep object
#     ps = ParameterSweep(**kwargs_dict)
#
#     results_array, results_dict = ps.parameter_sweep(
#         kwargs_dict["build_model"],
#         kwargs_dict["build_sweep_params"],
#         build_outputs=kwargs_dict["build_outputs"],
#         build_outputs_kwargs=kwargs_dict["build_outputs_kwargs"],
#         seed=None,
#         build_model_kwargs=kwargs_dict["build_model_kwargs"],
#         build_sweep_params_kwargs=kwargs_dict["build_sweep_params_kwargs"],
#     )
#     return results_array


if __name__ == "__main__":
    sh_pn_sweep_swro(nfe=2, save_results=True)
