from methods import InitialEnergy, ParticleTracer, InitialVelocity, Compute_B
import numpy as np
import pandas as pd
from scipy.constants import e, m_e, c, m_p
import time as time
import os

start = time.time()

# Constants
Re = 6378137                        # Earth radius in meters
B0 = 3.07e-5                        # Magnetic field at the equator in Tesla
B0_Re3 = B0 * (Re**3)               # Precompute B0 * Re^3
c2 = c**2                           # Speed of light squared

# Method
method = "Boris"

# Particle species
species = "proton"
if species == "proton":
    m = m_p
    q_sign = 1  # Positive charge
    qm_ratio = (q_sign*e) / m  # Charge-to-mass ratio
    filename = "Lorentz/proton/proton_trajectory"
    if not os.path.exists("Lorentz/proton"):
        os.makedirs("Lorentz/proton")
elif species == "electron":
    m = m_e
    q_sign = -1
    qm_ratio = (q_sign*e) / m
    filename = "Lorentz/electron/electron_trajectory"
    if not os.path.exists("Lorentz/electron"):
        os.makedirs("Lorentz/electron")
    
# Lorentz System
def lorentz_force(t, s):
    x, y, z, vx, vy, vz = s

    r2 = x**2 + y**2 + z**2
    r5 = r2**2.5
    v2 = vx**2 + vy**2 + vz**2
    gamma = 1 / np.sqrt(1 - v2 / c2)

    # Magnetic field components
    B_factor = -B0_Re3 / r5
    Bx = 3 * x * z * B_factor
    By = 3 * y * z * B_factor
    Bz = (2 * z**2 - x**2 - y**2) * B_factor

    # Derivatives
    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = qm_ratio * (vy * Bz - vz * By) / gamma
    dvydt = qm_ratio * (vz * Bx - vx * Bz) / gamma
    dvzdt = qm_ratio * (vx * By - vy * Bx) / gamma
    
    return np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt])

tracer = ParticleTracer(lorentz_force, species)

# Initial conditions
K = 1e7 * e                                                         # 10 MeV
v_mod = InitialVelocity(K, m)
print(f"v_mod = {v_mod/c:.5f} c")
print(f"K = {K/e:.1e} eV")

# Initial position: equatorial plane 4Re from Earth
x0, y0, z0 = 4 * Re, 0, 0

# Initial velocity
pitch_angle = 45. # degrees
vx0, vy0, vz0 = 0, v_mod * np.sin(np.radians(pitch_angle)), v_mod * np.cos(np.radians(pitch_angle))

# Initial conditions
initial_conditions = [x0, y0, z0, vx0, vy0, vz0]

dump = 1
t_span = [0, 40]
dt = 1e-3

# Solve the system using the Runge-Kutta 4th order method
S = tracer.solve(initial_conditions, t_span, method, dt, Bfield=Compute_B, dump=dump)
# tracer.plot(S, title="adsad")

time_array = np.arange(t_span[0], t_span[1], dt)[::dump]
dataframe = pd.DataFrame({'t': time_array, "x": S[:, 0], "y": S[:, 1], "z": S[:, 2], "vx": S[:, 3], "vy": S[:, 4], "vz": S[:, 5]})
dataframe.to_csv(f"{filename}_{K/e:.1e}_{int(pitch_angle)}_dt{dt:.1e}_{method}.csv", index=False)

tracer.save_to_html(S[:, 0]/Re, S[:, 1]/Re, S[:, 2]/Re, f"{filename}_{K/e:.1e}_{int(pitch_angle)}_dt{dt:.1e}_{method}.html")