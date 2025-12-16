import numpy as np
import matplotlib.pyplot as plt


def cp_induced_pressure_loss():
    feed_salinity = 5  # g/L
    feed_osm_pressure = 2 * 8.314 * 298 * (feed_salinity / 58.44 * 1000)  # Pa
    print(f"Feed osmotic pressure: {feed_osm_pressure/1e5:.2f} bar ")
    A = 5e-12  # m/(s·Pa), membrane permeability
    op_factor = 1  # operating pressure factor
    driving_pressure = op_factor * feed_osm_pressure * 0.99 # Assume 99% rejection
    Jw = A * driving_pressure  # m/s, water flux
    print(f"Water flux: {Jw*1e6:.2f} LMH ")
    D = 1e-9  # m^2/s, typical solute diffusivity (NaCl)
    d_h = 0.001  # m, characteristic length (hydraulic diameter)

    Sh_vals = np.logspace(1, 2, 100)

    cp_modulus = np.exp(Jw * d_h / (D * Sh_vals))
    print(f" Max CP modulus: {np.max(cp_modulus):.2f} ")
    print(f" Min CP modulus: {np.min(cp_modulus):.2f} ")
    print(f" Avg CP modulus: {np.mean(cp_modulus):.2f} ")
    dPi_CP = (cp_modulus - 1) * feed_osm_pressure
    print(f"Max CP-induced pressure: {np.max(dPi_CP)/1e5:.2f} bar ")
    print(f" Min CP-induced pressure: {np.min(dPi_CP)/1e5:.2f} bar ")
    print(f" Avg CP-induced pressure: {np.mean(dPi_CP)/1e5:.2f} bar ")


    rho = 1000  # kg/m3, water density
    mu = 1e-3  # Pa.s, dynamic viscosity
    L = 6  # m, length of the channel
    u = 0.25  # m/s, fixed feed velocity
    Re = (rho * u * d_h) / mu
    Pn_vals = np.logspace(5, 7, 100)
    print(f" Reynolds number: {Re:.1f} ")
    dPi_f = 0.5 * (L / d_h) * rho * u ** 2 * (2 * Pn_vals / Re ** 3)
    print(f"Max frictional pressure: {np.max(dPi_f)/1e5:.2f} bar ")
    print(f" Min frictional pressure: {np.min(dPi_f)/1e5:.2f} bar ")
    print(f" Avg frictional pressure: {np.mean(dPi_f)/1e5:.2f} bar ")

    axes = plt.subplots(1, 2, figsize=(14, 5))[1]
    axes[0].semilogx(Pn_vals, dPi_f / 1e5, label='Frictional Pressure Drop', color='blue')
    axes[0].set_xlabel('Power Number (Pn)')
    axes[0].set_ylabel('Pressure Drop (bar)')
    axes[0].set_title('Frictional Pressure Drop vs Power Number')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].semilogx(Sh_vals, dPi_CP / 1e5, label='Osmotic Back-Pressure', color='red')
    axes[1].set_xlabel('Sherwood Number (Sh)')
    axes[1].set_ylabel('Pressure Increase (bar)')
    axes[1].set_title('Osmotic Back-Pressure vs Sherwood Number')
    axes[1].legend()
    axes[1].grid(True)
    plt.show()


    total_dP = dPi_f[:, np.newaxis] + dPi_CP[np.newaxis, :]
    t_dP_ratio = total_dP * 1e5 / driving_pressure
    plt.figure(figsize=(8, 6))
    plt.contourf(Sh_vals, Pn_vals, t_dP_ratio / 1e5, levels=50, cmap='viridis')
    plt.colorbar(label='Total Pressure (bar)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sherwood Number (Sh)')
    plt.ylabel('Power Number (Pn)')
    plt.title('Total Pressure from Friction and Osmotic Effects')
    plt.show()



def compre_relative_energies_sd():
    # Constants and inputs
    rho = 1000       # kg/m3, water density
    mu = 1e-3        # Pa.s, dynamic viscosity
    A = 1.5e-11      # m/(s·Pa), membrane permeability
    d_h = 0.001      # m, hydraulic diameter
    L = 6            # m, channel length
    R = 0.5          # Recovery ratio
    u = 0.25         # m/s, feed velocity
    Jw = 5e-5        # m/s, water flux
    CP_modulus = 1.5 # dimensionless
    eta_pump = 0.85  # Pump efficiency

    # Osmotic pressure model
    k_osm = 7.7e4    # Pa per g/L
    S_feed = np.linspace(0, 100, 101)
    S_perm = 0.1
    pi_feed_ideal = k_osm * S_feed
    pi_perm = k_osm * S_perm

    # Reynolds number and pressure loss
    Re = (rho * u * d_h) / mu
    print(f"Reynolds number: {Re:.1f}")
    Pn = 1e6  # Power number

    def delta_P_loss(Pn, Re, L, d_h, rho, u):
        return (Pn / Re ** 3) * (L / d_h) * 0.5 * rho * u ** 2

    dP_loss = delta_P_loss(Pn, Re, L, d_h, rho, u)
    dP_membrane = Jw / A
    dPi_CP = (CP_modulus - 1) * pi_feed_ideal

    # Build arrays
    dP_loss_arr = np.full_like(S_feed, dP_loss)
    dP_membrane_arr = np.full_like(S_feed, dP_membrane)
    dP_pump = dP_membrane_arr + pi_feed_ideal + dPi_CP + dP_loss_arr

    # Energy input per m³ of permeate
    E = (1 / R) * dP_pump / eta_pump
    E_hydraulic = (1 / R) * dP_loss_arr / eta_pump
    E_osmotic_CP = (1 / R) * dPi_CP / eta_pump
    E_membrane = (1 / R) * dP_membrane_arr / eta_pump
    E_ideal_osmotic = (1 / R) * pi_feed_ideal / eta_pump

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(S_feed, E_hydraulic / 1e6, label='Hydraulic losses')
    plt.plot(S_feed, E_osmotic_CP / 1e6, label='Osmotic CP penalty')
    plt.plot(S_feed, E_membrane / 1e6, label='Membrane resistance')
    plt.plot(S_feed, E_ideal_osmotic / 1e6, label='Ideal osmotic pressure')
    plt.plot(S_feed, E / 1e6, label='Total energy', linewidth=2, color='black')
    plt.xlabel('Feed Salinity (g/L)')
    plt.ylabel('Energy per m³ permeate (MJ/m³)')
    plt.title('Energy Breakdown vs. Feed Salinity (Solution-Diffusion Model)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    compre_relative_energies_sd()
