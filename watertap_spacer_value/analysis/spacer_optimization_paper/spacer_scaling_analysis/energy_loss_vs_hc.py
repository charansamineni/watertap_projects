import numpy as np
import matplotlib.pyplot as plt

# Constants and parameters
Q = 0.2e-3  # m^3/s, feed flow rate
R = 0.5  # recovery ratio
C_bulk = 35.0  # g/L, feed salinity
Js_Jw = 0.0  # salt passage ratio (assumed negligible)

L = 6.0  # m, channel length
epsilon = 0.85  # channel porosity
rho = 1000.0  # kg/m3
mu = 1e-3  # Pa.s
k_osm = 7.7e4  # Pa·L/g

lambda_f1 = 0.42  # friction factor constant
lambda_f2 = 183  # friction factor coefficient
b = 0.005  # m, spacer width
D = 1e-9  # m^2/s, salt diffusivity
Sc = mu / (rho * D)  # Schmidt number
lambda_sh1 = 0.23  # Sherwood correlation constant

PF_1 = 0.004  # packing factor 1 (m^2)
PF_2 = (0.2032+0.254)*1e-3  # packing factor 2 (m)

# Range of channel heights in meters (0.1 mm to 1 mm)
hc = np.linspace(0.0001, 0.001, 200)

d_h = 2 * hc  # m, hydraulic diameter


# Compute membrane width W (m)
W = PF_1 / (hc + PF_2)

# Membrane area (m^2)
A_mem = L * W

# Water flux Jw (m/s)
Jw = (R * Q) / A_mem

# Axial velocity u (m/s)
u = Q / (epsilon * hc * W)

# Reynolds number Re
Re = (rho * u * d_h) / mu

# Exponential term argument for concentration polarization
exp_arg = (Jw * d_h) / (lambda_sh1 * (Re**0.33) * (Sc**0.33) * D)

# Osmotic pressure penalty (Pa)
delta_pi_cp = k_osm * (np.exp(exp_arg) - 1) * (C_bulk - Js_Jw)

# Pressure loss due to friction (Pa)
delta_P_loss = (rho / 2) * u**2 * (lambda_f1 + (lambda_f2 * mu) / (b * d_h * rho * u))

# Membrane pressure drop due to water flux (Pa)
# Assuming A (permeability) related as Jw = A * (dP - dPi)
# Rearranged: dP_membrane = Jw / A (use approximate permeability)
A_perm = 1.5e-11  # m/(s·Pa), typical permeability
dP_membrane = Jw / A_perm

# Bulk feed osmotic pressure (Pa)
pi_feed = k_osm * C_bulk

# Total pump pressure (Pa)
dP_pump = dP_membrane + pi_feed + delta_pi_cp + delta_P_loss

# Energy loss per volume of permeate (J/m^3)
E_loss = (1 / R) * dP_pump  # Assuming pump efficiency = 1 for simplicity

# Convert energy to MJ/m^3
E_loss_kWh = E_loss / 1e6 * 0.2778  # kWh/m^3
dP_bar = (delta_P_loss / L) / 1e5  # Convert Pa to bar

# Plot with two y-axes
fig, ax1 = plt.subplots(figsize=(8,6))

color1 = 'tab:blue'
ax1.set_xlabel('Channel Height $h_c$ (mm)')
ax1.set_ylabel('Energy Loss (kWh/m$^3$)', color=color1)
ax1.plot(hc * 1e3, E_loss_kWh, color=color1, label='Total Energy Loss')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Total Pressure (bar)', color=color2)
ax2.plot(hc * 1e3, dP_bar, color=color2, label='Pressure drop per unit length')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Energy Loss vs Channel Height for Seawater RO at 50% Recovery')
fig.tight_layout()
plt.grid(True)
plt.show()
