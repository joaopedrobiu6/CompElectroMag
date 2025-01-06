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
method = "RK4"

# Particle species
species = "electron"
if species == "proton":
    m = m_p
    q_sign = 1  # Positive charge
    qm_ratio = (q_sign*e) / m  # Charge-to-mass ratio
    filename = "GC/protons/proton_trajectory"
    if not os.path.exists("GC/protons"):
        os.makedirs("GC/protons")
elif species == "electron":
    m = m_e
    q_sign = -1
    qm_ratio = (q_sign*e) / m
    filename = "GC/electrons/electron_trajectory"
    if not os.path.exists("GC/electrons"):
        os.makedirs("GC/electrons")
    
# Guiding Center Equations
def getBmod(x, y, z):
    """
    Computes the magnitude of the magnetic field at a point (x, y, z) in a dipole field.
    """
    fac1 = -B0 * Re**3 / (x**2 + y**2 + z**2)**2.5
    Bx = 3 * x * z * fac1
    By = 3 * y * z * fac1
    Bz = (2 * z**2 - x**2 - y**2) * fac1
    return np.sqrt(Bx**2 + By**2 + Bz**2)

def GuidingCenter(t, s):
    
    x, y, z, vpar = s
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
    fac = m / (2 * e * B_mod**2) * (vsq + vpar**2)
    dxdt = fac * bxgB_x + vpar * b_unit_x
    dydt = fac * bxgB_y + vpar * b_unit_y
    dzdt = fac * bxgB_z + vpar * b_unit_z
    dvpar_dt = -mu / m * dotpr

    return [dxdt, dydt, dzdt, dvpar_dt]

tracer = ParticleTracer(GuidingCenter, species)

# Initial conditions
K = 1e7 * e                                                         # 10 MeV
v_mod = InitialVelocity(K, m)
print(f"v_mod = {v_mod/c:.5f} c")
print(f"K = {K/e:.1e} eV")

# Initial position: equatorial plane 4Re from Earth
x0, y0, z0 = 4 * Re, 0, 0
B_i = Compute_B(x0, y0, z0)

# Initial velocity
pitch_angle = 90. # degrees
vx0, vy0, vz0 = 0, v_mod * np.sin(np.radians(pitch_angle)), v_mod * np.cos(np.radians(pitch_angle))

vpar0= (vx0 * B_i[0] + vy0 * B_i[1] + vz0 * B_i[2]) / np.linalg.norm(B_i)

# Initial conditions
initial_conditions = [x0, y0, z0, vpar0]

dump = 1
t_span = [0, 1]
dt = 1e-4

# Solve the system using the Runge-Kutta 4th order method
S = tracer.solve(initial_conditions, t_span, method, dt, Bfield=Compute_B, dump=dump)
# tracer.plot(S, title="adsad")

time_array = np.arange(t_span[0], t_span[1], dt)[::dump]
dataframe = pd.DataFrame({'t': time_array, "x": S[:, 0], "y": S[:, 1], "z": S[:, 2], "vx": S[:, 3], "vy": S[:, 4], "vz": S[:, 5]})
dataframe.to_csv(f"{filename}_{K/e:.1e}_{int(pitch_angle)}_dt{dt:.1e}_{method}.csv", index=False)

tracer.save_to_html(S[:, 0]/Re, S[:, 1]/Re, S[:, 2]/Re, f"{filename}_{K/e:.1e}_{int(pitch_angle)}_dt{dt:.1e}_{method}.html")