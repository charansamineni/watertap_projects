from scipy.optimize import minimize
import numpy as np

# Constants
Q = 1.1e-1
mu = 1e-3
rho = 1000
L = 6
a = 0.42
b = 183

# Bounds for hc and eps
hc_bounds = (0.3e-3, 1.5e-3)
eps_bounds = (0.6, 0.95)

def f_to_optimize(vars):
    hc, eps = vars

    w = 180*0.04776103732 / (hc+0.254e-3+0.2032e-3)  # channel width

    # Compute dh
    dh = (4 * eps) / (2/hc + (1-eps)/(8*hc))

    # Compute u
    u = Q / (w * hc * eps)

    # Reynolds number
    Re = dh * u * rho / mu

    # Friction factor
    f = a + b / Re

    return f  # or -f for maximizing

# Find maximum f by minimizing -f
def print_vars_at_solution(sol):
    hc, eps = sol.x
    w = 180*0.004 / (hc+0.254e-3+0.2032e-3)
    dh = (4 * eps) / (2/hc + (1-eps)/(8*hc))
    u = Q / (w * hc * eps)
    Re = dh * u * rho / mu
    f_val = a + b / Re
    print(f"hc={hc:.2e}, eps={eps:.1f}, w={w:.1f}, dh={dh:.2e}, u={u:.2e}, Re={Re:.1f}, f={f_val:.1f}")


res_max = minimize(lambda x: -f_to_optimize(x), x0=[0.5e-3, 0.75],
                   bounds=[hc_bounds, eps_bounds])

print("Variables at maximum f:")
print_vars_at_solution(res_max)
res_min = minimize(f_to_optimize, x0=[0.5e-3, 0.75],
                   bounds=[hc_bounds, eps_bounds])
print("Variables at minimum f:")
print_vars_at_solution(res_min)
