import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import e, m_e, c, m_p
import pandas as pd
import plotly.graph_objects as go
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
time_array = []
# m = m_e
# q_sign = -1  # Negative charge
# qm_ratio = (q_sign*e) / m  # Charge-to-mass ratio

# Newton-Lorentz equation function
def Lorentz(t, y):
    # Unpack variables
    x, y, z, vx, vy, vz = y
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
    
    time_array.append(t)
    Bx_array.append(Bx)
    By_array.append(By)
    Bz_array.append(Bz)

    # Derivatives
    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = qm_ratio * (vy * Bz - vz * By) / gamma
    dvydt = qm_ratio * (vz * Bx - vx * Bz) / gamma
    dvzdt = qm_ratio * (vx * By - vy * Bx) / gamma
    
    return [dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt]

# Parameters for the trajectory
K = 5e6 * e  # Kinetic energy in Joules
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

# Time span
t_span = [0, 120]

# Solve the ODE
sol = solve_ivp(Lorentz, t_span, initial_conditions, method='RK45')#, rtol=1e-6, atol=1e-9)

# Extract solution
t = sol.t
x, y, z, vx, vy, vz = sol.y

Bx_array = np.array(Bx_array)
By_array = np.array(By_array)
Bz_array = np.array(Bz_array)
time_array = np.array(time_array)
indexes = np.searchsorted(time_array, t)
Bx_array = Bx_array[indexes]
By_array = By_array[indexes]
Bz_array = Bz_array[indexes]

dataframe = pd.DataFrame({'t': t, 'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz, 'Bx': Bx_array, 'By': By_array, 'Bz': Bz_array})
dataframe.to_csv(f"{filename}_{K/e:.1e}_{pitch_angle}.csv", index=False)

# Plot the trajectory with plotly
fig = go.Figure()
fig.add_trace(go.Scatter3d(x=x / Re, y=y / Re, z=z / Re, mode='lines', name='Trajectory'))
fig.update_layout(title='Proton trajectory', scene=dict(aspectmode='cube'), scene_aspectmode='cube')
fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
fig.update_layout(scene=dict(xaxis=dict(range=[-5, 5]), yaxis=dict(range=[-5, 5]), zaxis=dict(range=[-5, 5])))
# fig.write_html(f"{filename}_{K/e:.1e}_{pitch_angle}.html")

end = time.time()
print(f"Execution time: {end - start} s")