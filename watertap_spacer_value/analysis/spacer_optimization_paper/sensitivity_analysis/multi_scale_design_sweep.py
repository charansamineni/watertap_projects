from watertap_spacer_value.analysis.spacer_optimization_paper.sensitivity_analysis.sweep_builder import (
    sweep_build,
)
from watertap_spacer_value.analysis.spacer_optimization_paper.spacer_scale_benchmarks.literature_correlations.RO_case_studies_across_correlations import (
    solve,
    solve_for_recovery,
)
from parameter_sweep import LinearSample, ParameterSweep, GeomSample


def build_sweep_params(m, n_samples=11, design_type="module"):

    if design_type == "spacer":

        sweep_params = {
            "sh_multiplier": LinearSample(
                m.fs.ro[1].feed_side.Sh_improvement_factor, 0.5, 1.5, n_samples
            ),
            "friction_factor_multiplier": LinearSample(
                m.fs.ro[1].feed_side.friction_factor_improvement, 0.5, 1.5, n_samples
            ),
        }

    elif design_type == "system":

        sweep_params = {
            "modules_per_pv": LinearSample(
                m.fs.ro[1].length, 3, 9, n_samples
            ),
            "n_pv": LinearSample(
                m.fs.ro[1].n_pressure_vessels, 150, 210, n_samples
            ),
        }

    elif design_type == "module":

        sweep_params = {
            "channel_height": LinearSample(
                m.fs.ro[1].feed_side.channel_height, 23*2.54e-5, 33*2.54e-5, n_samples
            ),
            "spacer_porosity": LinearSample(
                m.fs.ro[1].feed_side.spacer_porosity, 0.75, 0.95, n_samples
            ),
        }
    else:
        raise ValueError("design_type must be either 'spacer' or 'system'")

    return sweep_params


def build_outputs(m):
    outputs = {
        "specific_energy_consumption": m.fs.costing.specific_energy_consumption,
        "sh_multiplier": m.fs.ro[1].feed_side.Sh_improvement_factor,
        "friction_factor_multiplier": m.fs.ro[1].feed_side.friction_factor_improvement,
        "LCOW": m.fs.costing.LCOW,
        "operating_pressure": m.fs.pump.outlet.pressure[0],
        "water_recovery": m.fs.water_recovery,
        "n_pv": m.fs.ro[1].n_pressure_vessels,
        "modules_per_pv": m.fs.ro[1].length,
        "channel_height": m.fs.ro[1].feed_side.channel_height,
        "spacer_porosity": m.fs.ro[1].feed_side.spacer_porosity,
        "velocity": m.fs.ro[1].feed_side.velocity[0, 0], # Inlet velocity
    }
    return outputs


def swro_design_sweep(
    n_samples,
    nfe=60,
    velocity=0.25,
    ro_system="SWRO",
    salinity=35,
    correlation_type="schock",
    sav_dir=".",
    design_type="spacer",
):
    import os

    # Create results directory based on ro_type and design_type
    folder_name = f"{ro_system}_nfe{nfe}_vel{velocity}_sal{salinity}_{correlation_type}_samples{n_samples}_design{design_type}"
    work_dir = os.path.join(sav_dir, folder_name)
    os.makedirs(work_dir, exist_ok=True)

    kwargs_dict = {
        "h5_results_file_name": os.path.join(work_dir, "sweep_results.h5"),
        "build_model": sweep_build,
        "build_model_kwargs": dict(
            nfe=nfe,
            velocity=velocity,
            ro_system=ro_system,
            salinity=salinity,
            correlation_type=correlation_type,
        ),
        "build_sweep_params": build_sweep_params,
        "build_sweep_params_kwargs": dict(n_samples=n_samples, design_type=design_type),
        "build_outputs": build_outputs,
        "build_outputs_kwargs": {},
        "optimize_function": solve_for_recovery,
        "optimize_kwargs": dict(recovery=0.5, display=False, strategy="simulation"),
        "initialize_function": solve,
        "initialize_kwargs": {},
        "parallel_back_end": "ConcurrentFutures",
        "number_of_subprocesses": 1,
        "csv_results_file_name": os.path.join(work_dir, "sweep_results.csv"),
        "h5_parent_group_name": None,
        "update_sweep_params_before_init": False,
        "initialize_before_sweep": False,
        "reinitialize_function": None,
        "reinitialize_kwargs": {},
        "reinitialize_before_sweep": False,
        "probe_function": None,
        "interpolate_nan_outputs": True,
    }

    # create parameter sweep object
    ps = ParameterSweep(**kwargs_dict)

    results_array, results_dict = ps.parameter_sweep(
        kwargs_dict["build_model"],
        kwargs_dict["build_sweep_params"],
        build_outputs=kwargs_dict["build_outputs"],
        build_outputs_kwargs=kwargs_dict["build_outputs_kwargs"],
        seed=None,
        build_model_kwargs=kwargs_dict["build_model_kwargs"],
        build_sweep_params_kwargs=kwargs_dict["build_sweep_params_kwargs"],
    )
    return results_array


if __name__ == "__main__":

    for design_type in ["module", "spacer", "system"]:
        if design_type == "system":
            n_samples = 7
        else:
            n_samples = 11
        for correlation in ["guillen", "schock", "dacosta", "koustou", "kuroda"]:
            swro_design_sweep(
                n_samples=n_samples,
                nfe=10,
                velocity=0.20,
                ro_system="SWRO",
                salinity=35,
                correlation_type=correlation,
                sav_dir=".",
                design_type=design_type,
            )
