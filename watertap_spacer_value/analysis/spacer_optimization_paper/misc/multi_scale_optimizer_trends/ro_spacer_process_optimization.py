import numpy as np
from watertap_spacer_value.flowsheets.ro_flowsheet_utils import *
import matplotlib.pyplot as plt
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.opt import assert_optimal_termination
from watertap.core.solvers import get_solver
from watertap_spacer_value.analysis.spacer_optimization_paper.spacer_scale_benchmarks.literature_correlations.RO_case_studies_across_correlations import sci_notation
from matplotlib import ticker as mticker
import matplotlib.patches as mpatches


def plot_multiple_strategies(strategies_to_plot=None):
    # Load results from Excel file
    results = {}
    macro_results = {}
    xls = pd.ExcelFile("ro_optimization_strategies_results.xlsx")
    for sheet_name in xls.sheet_names:
        if sheet_name.startswith("micro_"):
            strategy = sheet_name.replace("micro_", "")
            if strategy not in strategies_to_plot:
                continue
            df = pd.read_excel(xls, sheet_name=sheet_name)
            macro_df = pd.read_excel(xls, sheet_name="macro_trends", index_col=0)
            lcow = macro_df.loc[strategy, "LCOW"]
            results[strategy] = (df, lcow)
        elif sheet_name == "macro_trends":
            macro_df = pd.read_excel(xls, sheet_name=sheet_name, index_col=0)
            macro_results = macro_df

    # Plotting
    plt.rcParams.update({'font.size': 12, 'font.family': 'Arial'})
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))
    colors = {
        "topology": "#177e89",
        "simulation": "#084c61",
        "module": "#db3a34",
        "process": "#6a4c93",
        "topology_module_process": "#ff6f59",
    }
    font = {'family': 'Arial', 'size': 16}
    handles_labels = []

    for idx, (strategy, (df, lcow)) in enumerate(results.items()):
        color = colors[strategy]
        print(f"\nStrategy: {strategy} (LCOW={lcow:.3f})")
        for col in [
            'j_w', 'k_f', 'cp_penalty',
            'kinetic_energy', 'f', 'dp_friction'
        ]:
            if col in df:
                col_data = df[col]
                if col == "kinetic_energy":
                    min_val = col_data.min()
                    max_val = col_data.max()
                else:
                    min_val = col_data.iloc[1:].min()
                    max_val = col_data.iloc[1:].max()
                print(f"{col}: min={min_val:.3g}, max={max_val:.3g}")

        label_map = {
            "topology": "Spacer topology",
            "module": "Feed channel",
            "process": "Process",
            "simulation": "Base case",
            "topology_module_process": "Combined",
        }
        label_base = label_map.get(strategy, strategy)
        label = f"{label_base} optimization (LCOW={lcow:.2f} $/m³)" if strategy != "simulation" else f"{label_base} (LCOW={lcow:.2f} $/m³)"
        # Only collect the handle from the first plot for the legend
        h0, = axs[0, 0].plot(df['global_length_domain'][1:], df['j_w'][1:], label=label, color=color,
                             linestyle='--', linewidth=2, marker='o', markersize=5)
        axs[0, 1].plot(df['global_length_domain'][1:], df['k_f'][1:], color=color, linestyle='--', linewidth=2,
                       marker='o', markersize=5)
        axs[0, 2].plot(df['global_length_domain'][1:], df['cp_penalty'][1:], color=color, linestyle='--',
                       linewidth=2, marker='o', markersize=5)
        axs[1, 0].plot(df['global_length_domain'], df['kinetic_energy'], color=color, linestyle='--',
                       linewidth=2, marker='o', markersize=5)
        axs[1, 1].plot(df['global_length_domain'][1:], df['f'][1:], color=color, linestyle='--',
                       linewidth=2, marker='o', markersize=5)
        axs[1, 2].plot(df['global_length_domain'][1:], df['dp_friction'][1:], color=color, linestyle='--',
                       linewidth=2, marker='o', markersize=5)
        handles_labels.append((h0, label))

    # --- Macro results bar plots ---
    macro_strategies = [s for s in strategies_to_plot if s in results]
    macro_colors = [colors[s] for s in macro_strategies]
    macro_df = macro_results.loc[macro_strategies]

    # Area bar plot
    all_strategies = ["simulation", "topology", "module", "process", "topology_module_process"]
    axs[2, 0].bar(range(5), [macro_df["Total membrane area"].get(s, 0) for s in all_strategies],
                  color=[colors.get(s, "#cccccc") if s in macro_strategies else "#eeeeee" for s in all_strategies],
                  edgecolor='black')
    axs[2, 0].set_xticks(range(5))
    axs[2, 0].set_xticklabels([label_map.get(s, s).replace(" ", "\n") for s in all_strategies], fontdict=font)
    axs[2, 0].set_ylabel("Membrane area (m$^2$)", fontdict=font)
    axs[2, 0].set_ylim(0, 1.5e5)
    axs[2, 0].set_yticks(np.linspace(0, 1.5e5, num=11))
    axs[2, 0].yaxis.set_major_formatter(mticker.FuncFormatter(sci_notation))

    # Operating pressure bar plot
    axs[2, 1].bar(range(5), [macro_df["Operating pressure"].get(s, 0) / 1e5 if s in macro_df.index else 0 for s in
                             all_strategies],
                  color=[colors.get(s, "#cccccc") if s in macro_strategies else "#eeeeee" for s in all_strategies],
                  edgecolor='black')
    axs[2, 1].set_xticks(range(5))
    axs[2, 1].set_xticklabels([label_map.get(s, s).replace(" ", "\n") for s in all_strategies], fontdict=font)
    axs[2, 1].set_ylabel("Operating pressure (bar)", fontdict=font)
    axs[2, 1].set_ylim(0, 100)
    axs[2, 1].set_yticks(np.linspace(0, 100, num=11))

    cost_keys = ["levelized_mem_cost", "levelized_pump_cost", "levelized_operating_cost", "LCOW"]
    legend_labels = ["Membrane CapEX", "Pump CapEX", "OpEX", "LCOW"]
    hatch_patterns = ['|', '\\', '/', '']
    bar_width = 0.18
    x = np.arange(5)  # Always reserve space for 5 strategies


    bars = []
    bax = axs[2, 2]
    macro_x = np.array(
        [["simulation", "topology", "module", "process", "topology_module_process"].index(s) for s in macro_strategies])

    for i, (cost_key, hatch) in enumerate(zip(cost_keys, hatch_patterns)):
        bar = bax.bar(
            macro_x + i * bar_width,
            macro_df[cost_key],
            width=bar_width,
            color=macro_colors,
            edgecolor='black',
            hatch=hatch,
            label=legend_labels[i]
        )
        bars.append(bar)

    bax.set_xticks(np.arange(5) + 1.5 * bar_width)
    bax.set_xticklabels(
        [label_map.get(s, s).replace(" ", "\n") for s in
         ["simulation", "topology", "module", "process", "topology_module_process"]],
        fontdict=font
    )
    bax.set_xlim(-0.5, 5)

    # Y-label and axis break if needed for brokenaxes
    bax.set_ylabel("Cost (USD/m$^3$)", fontdict=font)
    bax.set_ylim(0, 1)
    bax.set_yticks(np.linspace(0, 1, num=11))

    # Custom legend: only hatch patterns, no face color
    legend_patches = [
        mpatches.Patch(
            facecolor='white',  # White face, not using macro_colors
            edgecolor='black',
            hatch=hatch_patterns[i],
            label=legend_labels[i]
        )
        for i in range(len(legend_labels))
    ]

    axs[2, 2].legend(
        handles=legend_patches,
        loc='upper right',
        fontsize=14,
        handlelength=2,
        handletextpad=1,
        fancybox=False,
        frameon=False,
        ncol=2
    )

    # --- Formatting ---
    for ax in axs.flat:
        ax.set_xlim(0, 10) if ax in axs[0, :] or ax in axs[1, :] else None
        if ax in axs[0, :] or ax in axs[1, :]:
            ax.set_xticks(np.linspace(0, 10, 11))

    axs[0, 0].set_xlabel("Axial position along the RO system (m)", fontdict=font)
    axs[0, 0].set_ylabel("Water flux (m/s)", fontdict=font)
    axs[0, 0].set_ylim(0, 2.5e-5)
    axs[0, 0].yaxis.set_major_formatter(mticker.FuncFormatter(sci_notation))
    axs[0, 0].set_yticks(np.linspace(0, 2.5e-5, num=11))

    axs[0, 1].set_xlabel("Axial position along the RO system (m)", fontdict=font)
    axs[0, 1].set_ylabel("Mass transfer coefficient (m/s)", fontdict=font)
    axs[0, 1].set_ylim(0,2.5e-4)
    axs[0, 1].yaxis.set_major_formatter(mticker.FuncFormatter(sci_notation))
    axs[0, 1].set_yticks(np.linspace(0, 2.5e-4, num=11))

    axs[0, 2].set_xlabel("Axial position along the RO system (m)", fontdict=font)
    axs[0, 2].set_ylabel("Osmotic back pressure (bar)", fontdict=font)
    axs[0, 2].set_ylim(0, 15)
    axs[0, 2].set_yticks(np.linspace(0, 15, num=11))

    axs[1, 0].set_xlabel("Axial position along the RO system (m)", fontdict=font)
    axs[1, 0].set_ylabel(r"Dynamic pressure (Pa) ($\frac{1}{2} \rho u^2$)", fontdict=font)
    axs[1, 0].set_ylim(0, 50)
    axs[1, 0].set_yticks(np.linspace(0, 50, num=11))

    axs[1, 1].set_xlabel("Axial position along the RO system (m)", fontdict=font)
    axs[1, 1].set_ylabel("Friction factor", fontdict=font)
    axs[1, 1].set_ylim(0, 2)
    axs[1, 1].set_yticks(np.linspace(0, 2, num=11))

    axs[1, 2].set_xlabel("Axial position along the RO system (m)", fontdict=font)
    axs[1, 2].set_ylabel("Spacer induced pressure drop (bar)", fontdict=font)
    axs[1, 2].set_ylim(0, 15)
    axs[1, 2].set_yticks(np.linspace(0, 15, num=11))

    for ax in axs.flat:
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname('Arial')
            label.set_fontsize(14)

    # Add a single legend at the bottom
    handles, labels = zip(*handles_labels)
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, prop=font,
               bbox_to_anchor=(0.5, 0.02))

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(
        f'ro_optimization_strategies_comparison_swro_{"".join([s[0] for s in strategies_to_plot])}_{len(strategies_to_plot)}.svg',
        dpi=300, bbox_inches='tight'
    )
    plt.show()


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
        m = init_build_swro_flowsheet(nfe=nfe, correlation_type=correlation_type)

        if s == "simulation":
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
            print(f" Electricity cost : {m.fs.costing.electricity_cost.value} USD/kWh")

        elif "process" not in s:
            if not hasattr(m.fs, "lcow_objective"):
                add_lcow_objective(m)

            unfix_variables(m, s)
            set_optimization_bounds(m)
            solve_for_recovery(m, recovery=0.5, tee=False, display=True)

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
            print(f" Electricity cost : {m.fs.costing.electricity_cost.value} USD/kWh")

        else:  # Process optimization involved
            if not hasattr(m.fs, "lcow_objective"):
                add_lcow_objective(m)
            unfix_variables(m, s)
            set_optimization_bounds(m)
            solve_for_recovery(m, recovery=0.5, tee=False, display=True)
            l_opt = m.fs.ro[1].length.value
            n_pv_opt = m.fs.ro[1].n_pressure_vessels.value
            possible_l = [np.floor(l_opt), np.ceil(l_opt)]
            possible_pv = [np.floor(n_pv_opt), np.ceil(n_pv_opt)]

            process_grid_macros = []
            process_grid_micro_df = {}
            for l in possible_l:
                for pv in possible_pv:
                    m.fs.ro[1].length.fix(l)
                    m.fs.ro[1].n_pressure_vessels.fix(pv)
                    try:
                        solve_for_recovery(m, recovery=0.5, tee=False, display=True)
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
                        macro_vars.update({"l": l, "pv": pv})
                        process_grid_macros.append(macro_vars)
                        process_grid_micro_df[f"{l}, {pv}"] = collect_micro_trend(m)
                    except Exception as e:
                        print(f"Skipping infeasible case l={l}, pv={pv}: {e}")

            process_grid_df = pd.DataFrame(process_grid_macros)
            best = min(process_grid_macros, key=lambda x: x["LCOW"])
            macro_trends_df[s] = {k: v for k, v in best.items() if k not in ["l", "pv"]}
            micro_trends_df[s] = process_grid_micro_df[f"{best['l']}, {best['pv']}"]
            print(f"Completed strategy: {s} with optimal LCOW = {macro_trends_df[s]['LCOW']:.4f}")
            process_grid_df.to_excel(f"process_grid_results_swro_{s}.xlsx", index=False)

    # Save the results to Excel
    macro_trends_df = pd.DataFrame(macro_trends_df).T
    print("\nMacro trends summary:")
    print(macro_trends_df.to_markdown(index=True))
    macro_trends_df["LCOW"] = pd.to_numeric(macro_trends_df["LCOW"], errors="coerce")
    opt_idx = macro_trends_df["LCOW"].idxmin()
    print(f"\nMost optimal design: {opt_idx} with LCOW = {macro_trends_df.loc[opt_idx, 'LCOW']:.4f}")
    with pd.ExcelWriter("ro_optimization_strategies_results_swro.xlsx") as writer:
        for s, df in micro_trends_df.items():
            df.to_excel(writer, sheet_name=f"micro_{s}", index=False)
        macro_trends_df.to_excel(writer, sheet_name="macro_trends", index=True)
    return


def unfix_variables(m, strategy):
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
    print(f"Degrees of freedom after using {strategy}: {degrees_of_freedom(m)}")
    return


def init_build_swro_flowsheet(nfe=60, correlation_type="schock"):
    m = build_swro_flowsheet(correlation_type=correlation_type, nfe=nfe)

    # Add telescoping potential variable and constraint to each stage
    for stage in m.fs.ro.values():
        add_telescoping_potential(stage)

    # Touch the osmotic pressure variable to ensure it is calculated
    for stage in m.fs.ro.values():
        for x in stage.feed_side.length_domain:
            stage.feed_side.properties[0, x].pressure_osm_phase[...]

    # Fix, scale, initialize, and solve the model
    fix_model(m, velocity=0.25, salinity=35, ro_system="SWRO")
    scale_model(m, ro_system="SWRO")
    print(f"Degrees of freedom after fixing: {degrees_of_freedom(m)}")
    initialize_model(m, overpressure=2.5, ro_system="SWRO")
    assert degrees_of_freedom(m) == 0, "DOF is not zero after initialization."
    solve(m, tee=False, display=False)
    add_costing(m)
    solve(m, tee=False, display=False)
    solve_for_recovery(m, recovery=0.5, tee=False, display=True)
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
    all_strategies = ["simulation"]
    results = run_multiple_strategies(all_strategies, nfe=10, correlation_type="schock")
    for i in range(2, len(all_strategies) + 1):
        strategies_to_plot = all_strategies[:i]
        plot_multiple_strategies(strategies_to_plot=strategies_to_plot)



