import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Global constants
B0 = 3.07e-5  # Tesla
Re = 6378137  # meter (Earth radius)
e = 1.602176565e-19  # Elementary charge (Coulomb)
m_pr = 1.672621777e-27  # Proton mass (kg)
m_el = 9.10938291e-31  # Electron mass (kg)
c = 299792458  # Speed of light (m/s)

# Choose particle properties (proton)
m = m_pr  # Replace with m_el for electron
q = e  # Replace with -e for negatively charged electron

# Proton trajectory properties
K = 1e7 * e  # Kinetic energy in Joules
v_mod = c / np.sqrt(1 + (m * c**2) / K)  # Speed
pitch_angle = 30.0  # degrees
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

    return [dxdt, dydt, dzdt, dvpar_dt]

# Initial conditions
x0 = 4 * Re
y0 = 0
z0 = 0
initial_conditions = [x0, y0, z0, v_par0]

# Time span
tfin = 80.0  # seconds
time = np.arange(0, tfin, 0.01)

# Solve the ODE
solution = solve_ivp(newton_lorentz_gc, [0, tfin], initial_conditions, t_eval=time, method='RK45')

# Extract solution
x_sol = solution.y.T  # Transpose for easier handling

# Plot trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_sol[:, 0] / Re, x_sol[:, 1] / Re, x_sol[:, 2] / Re, 'r')
ax.set_xlabel('x / Re')
ax.set_ylabel('y / Re')
ax.set_zlabel('z / Re')
ax.set_title('Guiding Center Trajectory')
plt.show()