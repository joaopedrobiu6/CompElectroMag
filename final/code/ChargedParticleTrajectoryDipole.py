import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Global variables
B0 = 3.07e-5  # Tesla
Re = 6378137  # meter (Earth radius)
e = 1.602176565e-19  # Elementary charge (Coulomb)
m_pr = 1.672621777e-27  # Proton mass (kg)
m_el = 9.10938291e-31  # Electron mass (kg)
c = 299792458  # Speed of light (m/s)
m = m_pr  # Proton mass
q = e  # Positive charge

# Newton-Lorentz equation function
def newton_lorentz(t, x_vect):
    """
    Function to compute the derivatives for the Newton-Lorentz equation.
    """
    # Unpack position and velocity
    x, y, z, u, v, w = x_vect

    # Magnetic field components
    fac1 = -B0 * Re**3 / (x**2 + y**2 + z**2)**2.5
    Bx = 3 * x * z * fac1
    By = 3 * y * z * fac1
    Bz = (2 * z**2 - x**2 - y**2) * fac1

    # Charge-to-mass ratio
    qom = q / m

    # Derivatives
    dxdt = u
    dydt = v
    dzdt = w
    dudt = qom * (v * Bz - w * By)
    dvdt = qom * (w * Bx - u * Bz)
    dwdt = qom * (u * By - v * Bx)

    return [dxdt, dydt, dzdt, dudt, dvdt, dwdt]

# Parameters for the proton trajectory
K = 1e7 * e  # Kinetic energy in Joules
v_mod = c / np.sqrt(1 + (m * c**2) / K)  # Speed

# Initial position: equatorial plane 4Re from Earth
x0 = 4 * Re
y0 = 0
z0 = 0

# Initial velocity
pitch_angle = 30.0  # degrees
u0 = 0.0
v0 = v_mod * np.sin(np.radians(pitch_angle))
w0 = v_mod * np.cos(np.radians(pitch_angle))

# Initial conditions
initial_conditions = [x0, y0, z0, u0, v0, w0]

# Time span
tfin = 80  # Final time in seconds
time = np.arange(0, tfin, 0.01)  # Time array

# Solve the ODE
solution = solve_ivp(newton_lorentz, [0, tfin], initial_conditions, t_eval=time, method='RK45')

# Extract the solution
x_sol = solution.y.T  # Transpose for easier handling (time steps in rows)

# Plotting the trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_sol[:, 0] / Re, x_sol[:, 1] / Re, x_sol[:, 2] / Re, 'r')

# Add a sphere at the center
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 1 * np.outer(np.cos(u), np.sin(v))
y = 1 * np.outer(np.sin(u), np.sin(v))
z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='b', alpha=0.9)


ax.set_xlabel('x/Re')
ax.set_ylabel('y/Re')
ax.set_zlabel('z/Re')
ax.set_title('Trajectory of a Proton in Dipole Magnetic Field')

# Set axis limits
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
ax.set_zlim([-4, 4])

plt.show()