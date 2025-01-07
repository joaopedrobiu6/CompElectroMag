import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from methods import InitialVelocity
from scipy.constants import e, m_e, m_p, c
import plotly.graph_objects as go

# Global constants
B0 = 3.07e-5  # Tesla
Re = 6378137  # meter (Earth radius)

# Particle species
species = "electron"
if species == "proton":
    m = m_p
    q_sign = 1  # Positive charge
    q = e
    qm_ratio = (q_sign*e) / m  # Charge-to-mass ratio
    filename = "GC/protons/proton_trajectory"
    if not os.path.exists("GC/protons"):
        os.makedirs("GC/protons")
elif species == "electron":
    m = m_e
    q_sign = -1
    q = -e
    qm_ratio = (q_sign*e) / m
    filename = "GC/electrons/electron_trajectory"
    if not os.path.exists("GC/electrons"):
        os.makedirs("GC/electrons")
    

# Proton trajectory properties
K = 1e7 * e  # Kinetic energy in Joules
v_mod = v_mod = InitialVelocity(K, m)

pitch_angle = 15  # degrees
v_par0 = v_mod * np.cos(np.radians(pitch_angle))  # Parallel velocity

# Magnetic field computation
def getBmod(x, y, z):
    """
    Computes the magnitude of the magnetic field at a point (x, y, z) in a dipole field.
    """
    fac1 = -B0 * Re**3 / (x**2 + y**2 + z**2)**2.5
    Bx = 3 * x * z * fac1
    By = 3 * y * z * fac1
    Bz = (2 * z**2 - x**2 - y**2) * fac1
    return np.sqrt(Bx**2 + By**2 + Bz**2)

# ODE function
def newton_lorentz_gc(t, x_vect):
    """
    Function defining the derivatives for the guiding center approximation.
    """
    x, y, z, vpar = x_vect
    vsq = v_mod**2
    fac1 = -B0 * Re**3 / (x**2 + y**2 + z**2)**2.5
    Bx = 3 * x * z * fac1
    By = 3 * y * z * fac1
    Bz = (2 * z**2 - x**2 - y**2) * fac1
    B_mod = np.sqrt(Bx**2 + By**2 + Bz**2)

    # Magnetic moment (adiabatic invariant)
    mu = m * (vsq - vpar**2) / (2 * B_mod)

    # Gradient of B_mod (numerical)
    d = 1e-3 * Re  # Small perturbation for finite difference
    gradB_x = (getBmod(x + d, y, z) - getBmod(x - d, y, z)) / (2 * d)
    gradB_y = (getBmod(x, y + d, z) - getBmod(x, y - d, z)) / (2 * d)
    gradB_z = (getBmod(x, y, z + d) - getBmod(x, y, z - d)) / (2 * d)

    # Unit vector along B
    b_unit_x = Bx / B_mod
    b_unit_y = By / B_mod
    b_unit_z = Bz / B_mod

    # Cross product b_unit x grad(B)
    bxgB_x = b_unit_y * gradB_z - b_unit_z * gradB_y
    bxgB_y = b_unit_z * gradB_x - b_unit_x * gradB_z
    bxgB_z = b_unit_x * gradB_y - b_unit_y * gradB_x

    # Dot product b_unit · grad(B)
    dotpr = b_unit_x * gradB_x + b_unit_y * gradB_y + b_unit_z * gradB_z

    # Guiding center velocity components
    fac = m / (2 * q * B_mod**2) * (vsq + vpar**2)
    dxdt = fac * bxgB_x + vpar * b_unit_x
    dydt = fac * bxgB_y + vpar * b_unit_y
    dzdt = fac * bxgB_z + vpar * b_unit_z
    dvpar_dt = -mu / m * dotpr

    return np.array([dxdt, dydt, dzdt, dvpar_dt])

# Initial conditions
x0 = 4 * Re
y0 = 0
z0 = 0
initial_conditions = [x0, y0, z0, v_par0]

# Time span
tfin = 200  # seconds
dt = 0.01
time = np.arange(0, tfin, dt)

# Solve the ODE
def RungeKutta4(f, t, x0):
    """
    Runge-Kutta 4th order method for solving ODEs.
    """
    n = len(t)
    x = np.zeros((n, len(x0)))
    x[0] = x0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        k1 = h * f(t[i], x[i])
        k2 = h * f(t[i] + h / 2, x[i] + k1 / 2)
        k3 = h * f(t[i] + h / 2, x[i] + k2 / 2)
        k4 = h * f(t[i] + h, x[i] + k3)
        x[i + 1] = x[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x

def solve(f, t, x0):
    """
    Solve ODE using Runge-Kutta 4th order method.
    """
    return RungeKutta4(f, t, x0)


# Extract solution
S = solve(newton_lorentz_gc, time, initial_conditions)

dataframe = pd.DataFrame({'t': time, "x": S[:, 0], "y": S[:, 1], "z": S[:, 2], "vpar": S[:, 3]})
dataframe.to_csv(f"{filename}_{K/e:.1e}_{int(pitch_angle)}_dt{dt:.1e}_GuidindCenter.csv", index=False)

# Plot trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(S[:, 0] / Re, S[:, 1] / Re, S[:, 2] / Re, 'r')
ax.set_xlabel('x / Re')
ax.set_ylabel('y / Re')
ax.set_zlabel('z / Re')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)
ax.set_title('Guiding Center Trajectory')
plt.show()

def save_to_html(x, y, z, filename):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x = x, y = y, z = z, mode='lines', name='Trajectory'))
    fig.update_layout(title='Proton trajectory', scene=dict(aspectmode='cube'), scene_aspectmode='cube')
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    fig.update_layout(scene=dict(xaxis=dict(range=[-5, 5]), yaxis=dict(range=[-5, 5]), zaxis=dict(range=[-5, 5])))
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_ = np.outer(np.cos(u), np.sin(v))
    y_ = np.outer(np.sin(u), np.sin(v))
    z_ = np.outer(np.ones(np.size(u)), np.cos(v))
    fig.add_trace(go.Surface(x=x_, y=y_, z=z_, colorscale='earth', showscale=False))
    fig.write_html(filename)
    
save_to_html(S[:, 0]/Re, S[:, 1]/Re, S[:, 2]/Re, f"{filename}_{K/e:.1e}_{int(pitch_angle)}_dt{dt:.1e}_GuidingCenter.html")