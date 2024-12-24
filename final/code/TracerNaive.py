from methods import ParticleTracer
import numpy as np
import pandas as pd
from scipy.constants import e, m_e, c, m_p
import time as time

start = time.time()
# define the proton and electron species


species = "electron"
if species == "proton":
    m = m_p
    q_sign = 1  # Positive charge
    qm_ratio = (q_sign*e) / m  # Charge-to-mass ratio
    filename = "proton_trajectory"
elif species == "electron":
    m = m_e
    q_sign = -1
    qm_ratio = (q_sign*e) / m
    filename = "electron_trajectory"

# Constants
B0_Re3 = 3.07e-5 * (6378137**3)  # Precompute B0 * Re^3
c2 = c**2  # Speed of light squared
Bx_array = []
By_array = []
Bz_array = []

def lorentz_force(t, s):
    # Unpack variables
    x, y, z, vx, vy, vz = s
    # Precompute shared terms
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
K = 1e7 * e  # Kinetic energy in Joules
v_mod = c / np.sqrt(1 + (m * c**2) / K)  # Speed

# Initial position: equatorial plane 4Re from Earth
Re = 6378137  # Earth radius in meters
x0, y0, z0 = 4 * Re, 0, 0

# Initial velocity
pitch_angle = 45.0  # degrees
vx0 = 0.0
vy0 = v_mod * np.sin(np.radians(pitch_angle))
vz0 = v_mod * np.cos(np.radians(pitch_angle))

# Initial conditions
initial_conditions = [x0, y0, z0, vx0, vy0, vz0]

# Solve the system using the Runge-Kutta 4th order method
S = tracer.solve(initial_conditions, [0, 20], "RungeKutta4", 1e-6)

time_array = np.arange(0, 20, 1e-6)[::4000]
print(f"Time shape: {time_array.shape}")
print(f"Shape: {S.shape}")

dataframe = pd.DataFrame({'t': time_array, "x": S[:, 0], "y": S[:, 1], "z": S[:, 2], "vx": S[:, 3], "vy": S[:, 4], "vz": S[:, 5]})
dataframe.to_csv(f"{filename}_{K/e:.1e}_{pitch_angle}.csv", index=False)

tracer.save_to_vtk_with_velocity(S[:, 0]/Re, S[:, 1]/Re, S[:, 2]/Re, S[:, 3]/c, S[:, 4]/c, S[:, 5]/c)
tracer.save_to_html(S[:, 0]/Re, S[:, 1]/Re, S[:, 2]/Re)
print(S)
tracer.plot(S)