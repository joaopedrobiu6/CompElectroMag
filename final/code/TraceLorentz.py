from methods import ParticleTracer, InitialVelocity, Compute_B
import numpy as np
import pandas as pd
from scipy.constants import e, m_e, c, m_p
import time as time
import os

start = time.time()
# define the proton and electron species
method = "Boris"

species = "electron"
if species == "proton":
    m = m_p
    q_sign = 1  # Positive charge
    qm_ratio = (q_sign*e) / m  # Charge-to-mass ratio
    filename = "new_results/protons/proton_trajectory"
    if not os.path.exists("new_results/protons"):
        os.makedirs("new_results/protons")
elif species == "electron":
    m = m_e
    q_sign = -1
    qm_ratio = (q_sign*e) / m
    filename = "new_results/electrons/electron_trajectory"
    if not os.path.exists("new_results/electrons"):
        os.makedirs("new_results/electrons")
    

# Constants
Re = 6378137                        # Earth radius in meters
B0 = 3.07e-5                        # Magnetic field at the equator in Tesla
B0_Re3 = B0 * (Re**3)               # Precompute B0 * Re^3
c2 = c**2                           # Speed of light squared

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

# Create the ParticleTracer object
tracer = ParticleTracer(lorentz_force, species)

# Initial conditions
K = 1e7 * e                                                         # Kinetic energy in Joules
v_mod = InitialVelocity(K, m)                                       # Speed

# Initial position: equatorial plane 4Re from Earth
x0, y0, z0 = 4 * Re, 0, 0

# Initial velocity
pitch_angle = 5. # degrees
vx0 = 0.0
vy0 = v_mod * np.sin(np.radians(pitch_angle))
vz0 = v_mod * np.cos(np.radians(pitch_angle))

# Initial conditions
initial_conditions = [x0, y0, z0, vx0, vy0, vz0]

dump = 1
t_span = [0, 10]
dt = 1e-5

# Solve the system using the Runge-Kutta 4th order method
S = tracer.solve(initial_conditions, t_span, method, dt, dump=dump)

time_array = np.arange(t_span[0], t_span[1], dt)[::dump]

dataframe = pd.DataFrame({'t': time_array, "x": S[:, 0], "y": S[:, 1], "z": S[:, 2], "vx": S[:, 3], "vy": S[:, 4], "vz": S[:, 5]})
dataframe.to_csv(f"{filename}_{K/e:.1e}_{int(pitch_angle)}_dt{dt:.1e}_{method}.csv", index=False)