import os
import pandas as pd
import json
import pint
from math import pi, log
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t

#########################################################################################################
# Global definitions
ureg = pint.UnitRegistry()
ureg.define("LMH = liter/ meter**2 / hour")
ureg.define("bar = 1e5 * pascal")
ureg.define("LMH_per_bar = LMH / bar")
# density and viscosity were accessed from https://wiki.anton-paar.com/us-en/water/
rho = 997 * ureg.kg / ureg.m ** 3
mu = 0.89e-3 * ureg.Pa * ureg.s
diffusivity = 1.49e-9 * ureg.m ** 2 / ureg.s
nacl_mw = 58.44 * ureg.g / ureg.mol
gas_constant = 8.314 * ureg.joule / (ureg.mole * ureg.kelvin)
temperature = 298 * ureg.kelvin
# Diffusivity is pulled from https://pubs.acs.org/doi/epdf/10.1021/ja01589a011?ref=article_openPDF for a concentration
# of approximately 5.15g/kg Nacl/H2O
mil_to_m = 2.54e-5



def perform_v_hc_fits():
    full_data = pd.read_excel('processed_data.xlsx')
    v_values = full_data['inlet_velocity'].unique()

    low_v_data = full_data[full_data['inlet_velocity'] == v_values[0]].copy()
    high_v_data = full_data[full_data['inlet_velocity'] == v_values[1]].copy()

    results_to_save = {}

    def perform_and_store_fit(y_variable, label):
        params, covariance = fit_correlation(
            y=y_variable,
            x=['inlet_velocity', 'channel_height'],
            function=k_v_h_fit,
            guess=[1, 1, 1],
            X_name="Inlet Velocity / Channel Height",
            Y_name=label,
            fit_df=full_data,
        )

        if params is None:
            print(f"Fit for {label} failed.")
            return None, None, None, None, None

        a, b, c = params
        t_value = t.ppf(0.975, df=len(full_data) - len(params))
        perr = np.sqrt(np.diag(covariance))

        # Compute fitted values and residuals
        for df in [low_v_data, high_v_data]:
            df[f'{label}_fitted'] = df.apply(
                lambda row: k_v_h_fitted(row['inlet_velocity'], row['channel_height'], a, b, c),
                axis=1
            )
            df[f'{label}_residual'] = df[y_variable] - df[f'{label}_fitted']

        # Compute goodness-of-fit
        actual = np.concatenate((low_v_data[y_variable].values, high_v_data[y_variable].values))
        fitted = np.concatenate((low_v_data[f'{label}_fitted'].values, high_v_data[f'{label}_fitted'].values))
        residuals = actual - fitted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        standard_error = np.sqrt(ss_res / (len(actual) - len(params)))

        # Save all results for Excel export
        results_to_save[label] = {
            "params": params,
            "perr": perr,
            "r_squared": r_squared,
            "standard_error": standard_error,
            "t_value": t_value
        }

        return params, perr, r_squared, standard_error, t_value

    # === Perform both fits ===
    ff_results = perform_and_store_fit("friction_factor", "friction_factor")
    k_results = perform_and_store_fit("mass_transfer_coeff", "k")

    plt.rcParams['figure.figsize'] = (12.0, 5.0)
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['xtick.minor.width'] = 1.2
    plt.rcParams['ytick.minor.width'] = 1.2
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    # === Combined figure with two subplots ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Friction factor subplot
    ax1.scatter(
        low_v_data['channel_height'] * 1e3, low_v_data['friction_factor'],
        color="indigo", marker="^", s=40, label="Velocity=0.1 m/s"
    )
    ax1.scatter(
        high_v_data['channel_height'] * 1e3, high_v_data['friction_factor'],
        color="indigo", marker="o", s=40, label="Velocity=0.4 m/s"
    )
    ax1.plot(
        low_v_data['channel_height'] * 1e3, low_v_data['friction_factor_fitted'],
        color='olive', linewidth=1.5, linestyle="--", label='Fitted friction factor'
    )
    ax1.plot(
        high_v_data['channel_height'] * 1e3, high_v_data['friction_factor_fitted'],
        color='olive', linewidth=2, linestyle="--"
    )
    ax1.set_ylabel("Friction factor")
    ax1.set_xlabel("Channel height (mm)")
    ax1.set_xlim(0.3, 0.65)
    ax1.set_ylim(0, 4.5)
    ax1.set_title("Friction factor correlation")
    ax1.legend()

    # Mass transfer coefficient subplot
    ax2.scatter(
        low_v_data['channel_height'] * 1e3, low_v_data['mass_transfer_coeff'],
        color="teal", marker="^", s=40, label="Velocity=0.1 m/s"
    )
    ax2.scatter(
        high_v_data['channel_height'] * 1e3, high_v_data['mass_transfer_coeff'],
        color="teal", marker="o", s=40, label="Velocity=0.4 m/s"
    )
    ax2.plot(
        low_v_data['channel_height'] * 1e3, low_v_data['k_fitted'],
        color='darkred', linewidth=1.5, linestyle="--", label='Fitted k'
    )
    ax2.plot(
        high_v_data['channel_height'] * 1e3, high_v_data['k_fitted'],
        color='darkred', linewidth=2, linestyle="--"
    )
    ax2.set_ylabel("Mass transfer coefficient (k)")
    ax2.set_xlim(0.3, 0.65)
    ax2.set_ylim(0, 3.5e-5)
    ax2.set_xlabel("Channel height (mm)")
    ax2.set_title("Mass transfer coefficient correlation")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("combined_fit_plots.png", dpi=300)
    plt.show()


    # === Save all to Excel ===
    with pd.ExcelWriter("fit_results_output.xlsx") as writer:
        low_v_data.to_excel(writer, sheet_name="Low Velocity", index=False)
        high_v_data.to_excel(writer, sheet_name="High Velocity", index=False)

        # Fit parameter summaries
        summary_rows = []
        for label, result in results_to_save.items():
            for i, param in enumerate(result["params"]):
                summary_rows.append({
                    "Correlation": label,
                    "Parameter": f"{['a','b','c'][i]}",
                    "Value": param,
                    "Std Error": result["perr"][i],
                    "95% CI Lower": param - result["t_value"] * result["perr"][i],
                    "95% CI Upper": param + result["t_value"] * result["perr"][i],
                })

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="Fit Parameters", index=False)

        # Fit metrics (R², standard error)
        metrics_df = pd.DataFrame([
            {
                "Correlation": label,
                "R_squared": result["r_squared"],
                "Standard_Error": result["standard_error"]
            } for label, result in results_to_save.items()
        ])
        metrics_df.to_excel(writer, sheet_name="Fit Metrics", index=False)

    print("All fits completed and saved to 'fit_results_output.xlsx'")


def create_re_fit_plots():
    df = process_raw_data()

    def sh_func_full(X, a, b):
        re = X[:, 0]
        sc = X[:, 1]
        return a * re ** b * sc ** 0.33

    def friction_func(X, a, b):
        re = X[:, 0]
        return a + b / re

    X_sh_full = df[["re", "schmidt_number"]].values
    y_sh = df["sherwood_number"].values
    popt_sh_full, pcov_sh_full = curve_fit(sh_func_full, X_sh_full, y_sh, p0=[0.5, 0.25])
    X_f = df[["re"]].values
    y_f = df["friction_factor"].values
    popt_f, pcov_f = curve_fit(friction_func, X_f, y_f, p0=[0.01, 1.0])

    re_range = np.linspace(df["re"].min(), df["re"].max(), 100)
    sc_val = df["schmidt_number"].iloc[0]
    X_plot_sh = np.column_stack([re_range, np.full_like(re_range, sc_val)])
    sh_fit_full = sh_func_full(X_plot_sh, *popt_sh_full)
    X_plot_f = re_range.reshape(-1, 1)
    f_fit = friction_func(X_plot_f, *popt_f)

    # Plotting
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 12

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))


    # Sherwood number fit
    axs[0].scatter(df["re"], df["sherwood_number"], label="Data", color="teal")
    axs[0].plot(re_range, sh_fit_full, "r--",
                label=f"Fit: a={popt_sh_full[0]:.2f}, b={popt_sh_full[1]:.2f}, sc^0.33")
    axs[0].set_xlabel("Reynolds number")
    axs[0].set_ylabel("Sherwood number")
    axs[0].legend()
    axs[0].set_title("Sherwood number empirical fit")

    # Friction factor fit
    axs[1].scatter(df["re"], df["friction_factor"], label="Data", color="indigo")
    axs[1].plot(re_range, f_fit, "g--", label=f"Fit: a={popt_f[0]:.2e}, b={popt_f[1]:.2e}")
    axs[1].set_xlabel("Reynolds number")
    axs[1].set_ylabel("Friction factor")
    axs[1].legend()
    axs[1].set_title("Friction factor empirical fit")

    plt.tight_layout()
    plt.savefig("re_fit_plot.png", dpi=300)
    plt.show()


# Model function: y = v^a * (b * hc + c)
def k_v_h(v, hc, a, b, c):
    return (v ** a) * (b * hc + c)

# Helper for curve_fit: expects X as (2, N) array
def k_v_h_fit(X, a, b, c):
    v, hc = X
    return k_v_h(v, hc, a, b, c)

# Fit correlation for a given y-variable
def fit_correlation(y, x, fit_df, function, guess, X_name, Y_name):
    X = np.vstack([fit_df[x[0]].values, fit_df[x[1]].values])
    y_data = fit_df[y].values
    try:
        params, covariance = curve_fit(function, X, y_data, p0=guess, maxfev=10000)
        return params, covariance
    except Exception as e:
        print(f"Fit for {Y_name} failed: {e}")
        return None, None

# Fitted value for a single row
def k_v_h_fitted(v, hc, a, b, c):
    return k_v_h(v, hc, a, b, c)



def process_raw_data(data_path="CFD_campaign_results.xlsx", data_sheet="python_input",
                     fixed_var_path="spacer_geometry.json"):
    global mil_to_m
    full_data = {}  # convert all the data to SI units
    geom = load_geometry(fixed_var_path)
    data = load_data_from_excel(file_name=data_path, sheet_name=data_sheet)
    full_data["spacer_height"] = data["Spacer_height(mil)"] * mil_to_m
    full_data["channel_height"] = (
            full_data["spacer_height"] / geom["spacer_hs_hc_ratio"].magnitude
    )
    full_data_df = pd.DataFrame(full_data)
    full_data_df["spacer_diameter"] = [
                                          (geom["spacer_diameter"].to(ureg.m)).magnitude
                                      ] * len(full_data_df)
    full_data_df["spacer_spacing"] = [
                                         (geom["spacer_spacing"].to(ureg.m)).magnitude
                                     ] * len(full_data_df)
    full_data_df[["spacer_porosity", "dh"]] = full_data_df.apply(
        lambda row: dome_porosity_dh(
            row["spacer_height"],
            row["channel_height"],
            row["spacer_diameter"],
            row["spacer_spacing"],
        ),
        axis=1,
        result_type="expand",
    )
    full_data_df["inlet_velocity"] = data["Inlet_velocity(m/s)"]
    full_data_df["re"] = full_data_df.apply(
        lambda row: reynolds(
            row["inlet_velocity"],
            row["dh"],
        ),
        axis=1,
        result_type="expand",
    )
    full_data_df["dP/dx"] = (
            data["deltaP(Pa)"] / geom["channel_length"].to(ureg.m).magnitude
    )
    full_data_df["friction_factor"] = full_data_df.apply(
        lambda row: friction_factor(
            row["dP/dx"],
            row["dh"],
            row["inlet_velocity"],
        ),
        axis=1,
    )
    full_data_df["feed_pressure"] = [
                                        geom["operational_pressure"].to(ureg.Pa).magnitude
                                    ] * len(full_data_df)
    full_data_df["permeate_pressure"] = [
                                            geom["operational_permeate_pressure"].to(ureg.Pa).magnitude
                                        ] * len(full_data_df)
    full_data_df["feed_molarity"] = [
                                        geom["operational_feed_conc"].to(ureg.mol / ureg.L).magnitude
                                    ] * len(full_data_df)
    full_data_df["osm_pressure"] = full_data_df.apply(
        lambda row: calc_osm_pressure(row["feed_molarity"]),
        axis=1,
    )
    full_data_df["cp_modulus"] = data["C/Cinit(average)"]
    full_data_df["jw"] = full_data_df.apply(
        lambda row: j_w(
            row["feed_pressure"],
            row["permeate_pressure"],
            row["feed_molarity"],
            row["cp_modulus"],
            row["dP/dx"],
            geom["channel_length"].to(ureg.m).magnitude,
            params=geom,
        ),
        axis=1,
    )
    full_data_df["mass_transfer_coeff"] = full_data_df.apply(
        lambda row: K(row["cp_modulus"], row["jw"]),
        axis=1,
    )
    full_data_df["sherwood_number"] = full_data_df.apply(
        lambda row: sherwood_number(row["mass_transfer_coeff"], row["dh"]),
        axis=1,
    )
    full_data_df["schmidt_number"] = [schmidt()] * len(full_data_df)

    return full_data_df


def load_geometry(file_name=None):
    with open(file_name, "r") as file:
        geometry = json.load(file)
    variables = {}
    for key in geometry.keys():
        for param in geometry[key]:
            var_name = f"{key}_{param['name']}"
            value = param["value"]
            unit = param["unit"]
            if param["name"] == "feed_conc":
                quantity = calculate_molarity(value, unit)
            else:
                quantity = value * ureg(unit)
            variables[var_name] = quantity
            exec(f"{var_name} = quantity")
            # print(f"{var_name}: {variables[var_name].to_base_units()}")
    return variables


def load_data_from_excel(file_name=None, sheet_name="python_input"):
    directory = os.getcwd()
    file_path = os.path.join(directory, file_name)
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df


def dome_porosity_dh(hs, hc, ds, ss):
    spherical_radius = ((0.5 * ds) ** 2 + hs ** 2) / (2 * hs)
    unit_spacer_area = 2 * pi * spherical_radius * hs
    unit_spacer_volume = pi * hc * (3 * (0.5 * ds) ** 2 + hs ** 2) / 6
    spacer_density = 1 / (ss ** 2)
    spacer_porosity = 1 - ((spacer_density * unit_spacer_volume) / hc)
    dh = (
            4
            * spacer_porosity
            / (2 / hc + (1 - spacer_porosity) * unit_spacer_area / unit_spacer_volume)
    )
    return spacer_porosity, dh


def friction_factor(dp_dx, dh, v, density=None):
    if density is None:
        global rho
        density = rho.to_base_units().magnitude

    f_value = (2 * dp_dx * dh) / (density * v ** 2)
    return f_value


def schmidt(density=None, viscosity=None, D=None):
    if density is None and viscosity is None and D is None:
        global rho
        global mu
        global diffusivity
        density = rho
        viscosity = mu
        D = diffusivity

    sc = (viscosity / (density * D)).to_base_units().magnitude
    return sc


def sherwood_number(k, dh):
    global diffusivity
    sh = (k * dh) / (diffusivity.to_base_units()).magnitude
    return sh


def calculate_molarity(mass, unit):
    global nacl_mw
    global rho

    molarity = (mass * ureg(unit) / nacl_mw) / (1 * ureg.kg / rho)
    return molarity


def calc_osm_pressure(molar_conc):
    global gas_constant
    global temperature
    i = 2  # Van't Hoff factor for NaCl (Na+ and Cl-)
    osm_pressure = (
        (i * molar_conc * (ureg.mol / ureg.L) * gas_constant * temperature).to(ureg.Pa)
    ).magnitude
    return osm_pressure


def j_w(feed_pressure, permeate_pressure, feed_molarity, cp_modulus, dp_dx, channel_length, params=None):
    global gas_constant, temperature
    a_value = (params["membrane_A"].to_base_units()).magnitude

    i = 2  # Van’t Hoff factor for NaCl

    # Convert molarity to mol/m³ to ensure pressure ends up in Pascals
    concentration = (feed_molarity * cp_modulus * ureg.mol / ureg.L).to(ureg.mol / ureg.meter**3)

    osmotic_pressure = (i * concentration * gas_constant * temperature).to(ureg.Pa).magnitude

    friction_delta_p = dp_dx * channel_length * 0.5  # assuming linear drop, take average pressure

    jw = a_value * ((feed_pressure - permeate_pressure - friction_delta_p) - osmotic_pressure)
    return jw



def K(cp, jw):
    k_value = jw / log(cp)
    return k_value


def reynolds(v, dh):
    global rho, mu
    re = (rho.magnitude * v * dh) / mu.magnitude
    return re


def export_data():
    full = process_raw_data()
    with pd.ExcelWriter("processed_data.xlsx") as report:
        full.to_excel(report, sheet_name="all_vars_calculated", index=True)


if __name__ == "__main__":
    perform_v_hc_fits()
