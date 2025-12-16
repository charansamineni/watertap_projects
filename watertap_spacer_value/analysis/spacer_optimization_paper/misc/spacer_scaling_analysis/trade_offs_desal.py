import numpy as np
from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
from pyomo.environ import ConcreteModel
from idaes.core import FlowsheetBlock
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def sh_pn_trade_off():
    # --- System constants ---
    A = 2.78e-7  # m/s/bar
    hc = 0.7112e-3  # m
    eps = 0.85
    dh = (4 * eps) / (2 / hc + (8 * (1 - eps) / hc))
    l_mem = 6 # m
    rec = 0.5  # Recovery
    j_w = 100 * 2.78e-7 # LMH

    # --- Inputs ---
    sh_values = np.logspace(np.log10(10), np.log10(200), 20)
    pn_values = np.logspace(5, 7, 20)
    SH, PN = np.meshgrid(sh_values, pn_values, indexing='ij')  # meshgrid with SH on rows

    # --- Property function ---
    c_f = 35  # g/L
    dens, mu, osm_p, diff = generate_props(np.array([c_f]))
    dens = dens[0]
    mu = mu[0]
    osm_p = osm_p[0] * 1e-5  # bar
    diff = diff[0]

    # --- Calculations ---
    exp_arg = (j_w * dh)/ (SH * diff)
    cp = np.exp(exp_arg)
    delta_pi_cp = osm_p * (np.exp(exp_arg) - 1)  # bar
    print(f"Max CP penalty: {np.max(delta_pi_cp):.2f} bar")
    print(f"Min CP penalty: {np.min(delta_pi_cp):.2f} bar")

    ###
    # j_w = Q_f * R / A_mem
    # u = Q_f / (eps * hc * W)
    # W = A_mem / l_mem
    # u = (j_w * l_mem) / (eps * hc * R)
    u = (j_w * l_mem) / (eps * hc * rec) * 0.75 # m/s # derate by 25% to account for non-uniform flow
    re = (dens * u * dh) / mu
    f = 2 * PN/(re**3)
    delta_p_friction = (dens / 2) * u ** 2 * f * (l_mem / dh) * 1e-5  # bar
    print(f"Max friction loss: {np.max(delta_p_friction):.2f} bar")
    print(f"Min friction loss: {np.min(delta_p_friction):.2f} bar")

    p_required_ideal = j_w / A + osm_p # bar
    print(f"Max total pressure: {np.max(p_required_ideal):.2f} bar")
    print(f"Min total pressure: {np.min(p_required_ideal):.2f} bar")
    p_required_with_losses = p_required_ideal + delta_pi_cp + delta_p_friction
    print(f"Max total pressure with losses: {np.max(p_required_with_losses):.2f} bar")
    print(f"Min total pressure with losses: {np.min(p_required_with_losses):.2f} bar")

    fig, axs = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    # 1. Contour for CP penalty
    levels = np.linspace(0, 80, 11)
    cp1 = axs[0].contourf(sh_values, pn_values, delta_pi_cp.T, levels=levels, cmap='viridis')
    fig.colorbar(cp1, ax=axs[0], label='CP Penalty (bar)', ticks= levels, format='%.1f')
    axs[0].set_title('Concentration Polarization Penalty')
    axs[0].set_xlabel('Sherwood Number, $Sh$')
    axs[0].set_ylabel('Power Number, $Pn$')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')

    # 2. Contour for friction losses
    levels = np.linspace(0, 4, 11)
    cp2 = axs[1].contourf(sh_values, pn_values, delta_p_friction.T, levels=levels, cmap='viridis')
    fig.colorbar(cp2, ax=axs[1], label='Friction losses (bar)', ticks=levels, format='%.1f')
    axs[1].set_title('Friction Loss')
    axs[1].set_xlabel('Sherwood Number, $Sh$')
    axs[1].set_ylabel('Power Number, $Pn$')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')

    # 3. Contour for total required pressure with losses
    levels = np.linspace(120, 210, 11)
    cp3 = axs[2].contourf(sh_values, pn_values, p_required_with_losses.T, levels=levels, cmap='viridis')
    fig.colorbar(cp3, ax=axs[2], label='Total pressure (bar)',ticks=levels, format='%.1f')
    axs[2].set_title('Total Required Pressure (with losses)')
    axs[2].set_xlabel('Sherwood Number, $Sh$')
    axs[2].set_ylabel('Power Number, $Pn$')
    axs[2].set_xscale('log')
    axs[2].set_yscale('log')

    plt.suptitle(f'RO operational trade-offs for {c_f:.1f}g/L, {j_w/2.78e-7:.1f}LMH/bar', fontsize=14)
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'font.family': 'Arial'})
    plt.savefig(f'sh_pn_tradeoff_desal_{c_f:.1f}gL_{j_w/2.78e-7:.1f}LMH.png')
    plt.show()



def jw_c_trade_off():
    # --- System constants ---
    hc = 0.7112e-3  # m
    eps = 0.85
    dh = (4 * eps) / (2 / hc + (8 * (1 - eps) / hc))
    l_mem = 6 # m
    rec = 0.5  # Recovery

    # --- Inputs ---
    j_lmh = np.linspace(10, 100, 10)  # Water flux: 1 to 100 LMH
    c_f = np.linspace(5, 75, 15)  # Feed conc: 1 to 100 g/L
    J, C = np.meshgrid(j_lmh, c_f, indexing='ij')  # meshgrid with J on rows
    J = J * 2.78e-7  # Convert to m/s

    # --- Property function ---
    # Assumes generate_props returns (density, viscosity, osmotic_pressure, diffusivity)
    dens, mu, osm_p, diff = generate_props(C.flatten())
    dens = dens.reshape(J.shape)
    mu = mu.reshape(J.shape)
    osm_p = osm_p.reshape(J.shape)
    diff = diff.reshape(J.shape)


    # # --- Calculations ---
    u = (J * l_mem) / (eps * hc * rec)  # m/s
    sc = mu / (dens * diff)
    re = (dens * u * dh) / mu
    sh = 0.46 * (re * sc) ** 0.36
    f = 0.42 + (183 / re)
    k_f = sh * diff / dh
    exp_arg = J / k_f
    print(f"Max exp arg: {np.max(exp_arg):.2f}")
    print(f"Min exp arg: {np.min(exp_arg):.2f}")
    cp = np.exp(exp_arg)
    print(f"Max cp: {np.max(cp):.2f}")
    print(f"Min cp: {np.min(cp):.2f}")
    dp_cp_bar = osm_p * (cp - 1) * 1e-5  # bar
    print(f"Max CP penalty: {np.max(dp_cp_bar):.2f} bar")
    print(f"Min CP penalty: {np.min(dp_cp_bar):.2f} bar")
    dp_f_bar = (dens / 2) * u ** 2 * f * (l_mem / dh) * 1e-5  # bar
    print(f"Max friction loss: {np.max(dp_f_bar):.2f} bar")
    print(f"Min friction loss: {np.min(dp_f_bar):.2f} bar")
    A_high = 6 * 2.78e-7  # m/s/bar
    A_low = 1 * 2.78e-7  # m/s/bar
    A = np.where(C < 20, A_high, A_low)
    p_required = (J / A) + osm_p * 1e-5
    print(f"Max total pressure: {np.max(p_required):.2f} bar")
    print(f"Min total pressure: {np.min(p_required):.2f} bar")
    p_req_vs_dp_ratio = (dp_cp_bar+dp_f_bar) / p_required
    print(f"Max P_req / (DP_cp + DP_f): {np.max(p_req_vs_dp_ratio):.2f}")
    print(f"Min P_req / (DP_cp + DP_f): {np.min(p_req_vs_dp_ratio):.2f}")

    # --- Prepare for seaborn heatmap ---
    # Create DataFrame with flux as rows (y-axis), feed conc as columns (x-axis), sorted low to high
    df = pd.DataFrame(p_req_vs_dp_ratio, index=np.round(j_lmh, 2), columns=np.round(c_f, 2))
    df = df.sort_index(axis=0, ascending=False)  # Sort flux (y-axis) low to high
    df = df.sort_index(axis=1, ascending=True)  # Sort feed conc (x-axis) low to high
    # --- Plot heatmap ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap='viridis',
                annot=True, fmt=".2f",
                vmin=0, vmax=0.60,
                cbar_kws={'label': r'$\Delta P / P$', 'ticks': np.linspace(0, 0.60, 11)},)
    plt.title(r'Trade-off between Water Flux and Feed Concentration for RO at 50% Recovery')
    plt.xlabel('Feed Concentration, $C_{feed}$ (g/L)')
    plt.ylabel('Water Flux, $J_w$ (LMH)')
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'font.family': 'Arial'})
    plt.tight_layout()
    plt.savefig(f'jw_c_tradeoff_desal.png')
    plt.show()



def generate_props(c_list):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.params = SeawaterParameterBlock()
    m.fs.properties = m.params.build_state_block([0], defined_state=True)
    m.fs.properties[0].temperature.fix(298.15)
    m.fs.properties[0].pressure.fix(1e5)
    m.fs.properties[0].conc_mass_phase_comp[...]
    m.fs.properties[0].pressure_osm_phase[...]
    m.fs.properties[0].visc_d_phase[...]
    m.fs.properties[0].diffus_phase_comp[...]

    dens_values = []
    mu_values = []
    osm_p_values = []
    diff_values = []
    for c in c_list:
        m.fs.properties.calculate_state(
            var_args={
                (
                    "conc_mass_phase_comp",
                    ("Liq", "TDS"),
                ): c,  # feed mass concentration
                ("flow_vol_phase", "Liq"): 1,
            },  # volumetric feed flowrate [-]
            hold_state=True,  # fixes the calculated component mass flow rates
        )
        dens_values.append(m.fs.properties[0].dens_mass_phase["Liq"].value)
        mu_values.append(m.fs.properties[0].visc_d_phase["Liq"].value)
        osm_p_values.append(m.fs.properties[0].pressure_osm_phase["Liq"].value)
        diff_values.append(m.fs.properties[0].diffus_phase_comp["Liq", "TDS"].value)

        # Reset state
        m.fs.properties[0].flow_mass_phase_comp[...].unfix()

    return np.array(dens_values), np.array(mu_values), np.array(osm_p_values), np.array(diff_values)


if __name__ == "__main__":
    pass


