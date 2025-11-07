import numpy as np
import matplotlib.pyplot as plt

# Non-dimensional parameter ranges
Re_vals = np.linspace(100, 200, 100)  # Reynolds number along the membrane
Sh_vals = np.linspace(10, 200, 100)   # Sherwood numbers
Pn_vals = np.linspace(1e5, 1e7, 100)  # Power number range

# Assumed constants
Jw = 5e-5  # m/s, typical RO water flux (~50 LMH)
D = 1e-9   # m^2/s, typical solute diffusivity (NaCl)
L = 0.001   # m, characteristic length (hydraulic diameter)
rho = 1000  # kg/m^3, water density

# Compute pressure drop due to friction (from Power number)
# Power number: Pn = 0.5 * f * Re^3 => f = 2*Pn / Re^3
# Darcy-Weisbach: DeltaP = f/2 * rho * u^2 / dh
# Approximate velocity from Re: u = Re * nu / L
nu = 1e-6  # kinematic viscosity, m^2/s
dh = L     # hydraulic diameter, m

# We'll use mean Re for plotting vs Pn
Re_mean = 150
u = Re_mean * nu / L
friction_drops = []
R = 8.314  # J/(mol·K)
T = 298  # K
salinity = 35 # g/L
M = 58.44  # g/mol for NaCl
C = salinity / M * 1000  # mol/m^3
osmotic_pressure = 2 * R * T * C  # Pa


for Pn in Pn_vals:
    f = 2 * Pn / (Re_mean**3)
    deltaP_fric = (f / 2) * rho * u**2 / dh
    friction_drops.append(deltaP_fric)

# Compute osmotic back-pressure via concentration polarization
# cm ~ exp(Jw / kf) ~ exp(Jw * L / (D * Sh))
osmotic_effects = np.exp(Jw * L / (D * Sh_vals))  # dimensionless concentration amplification
#
# # Normalize for plotting comparison
# friction_drops_norm = (friction_drops - np.min(friction_drops)) / (np.max(friction_drops) - np.min(friction_drops))
# osmotic_effects_norm = (osmotic_effects - np.min(osmotic_effects)) / (np.max(osmotic_effects) - np.min(osmotic_effects))
#
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#
# # Plot 1: Frictional Pressure Drop vs Power Number
# axes[0].plot(Pn_vals, friction_drops_norm, label='Frictional Pressure Drop (normalized)', color='blue')
# axes[0].set_xlabel('Power Number (Pn)')
# axes[0].set_ylabel('Normalized Effect')
# axes[0].set_title('Frictional Pressure Drop vs Power Number')
# axes[0].legend()
# axes[0].grid(True)
#
# # Plot 2: Osmotic Back-Pressure vs Sherwood Number
# axes[1].plot(Sh_vals, osmotic_effects_norm, label='Osmotic Back-Pressure (normalized)', color='red')
# axes[1].set_xlabel('Sherwood Number (Sh)')
# axes[1].set_ylabel('Normalized Effect')
# axes[1].set_title('Osmotic Back-Pressure vs Sherwood Number')
# axes[1].legend()
# axes[1].grid(True)
#
# fig.tight_layout()
# plt.show()
