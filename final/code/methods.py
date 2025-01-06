import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, c, m_p
import vtk
import plotly.graph_objects as go


Re = 6378137  # Earth radius in meters
B0 = 3.12e-5  # Magnetic field at the equator in Tesla
B0_Re3 = B0 * Re**3  # Precompute B0 * Re^3
"""
Particle Tracer Class

Takes as input the system to solve as a function f(t, s) = dsdt where s is the state [x, y, z, vx, vy, vz] that returns dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt

The class has different methods to solve the system and plot the results.
- Runge Kutta 4th order
- Euler's method
- Verlet method
- Leapfrog method

One method is "solve" that takes as input the initial conditions and the time span to solve the system and the method to use as a string.
"""

class ParticleTracer:
    def __init__(self, system, species):
        self.system = system
        self.species = species
        
        if species == "proton":
            self.m = m_p
            self.q_sign = 1
            self.qm_ratio = (self.q_sign*e) / self.m
            self.filename = "proton_trajectory"
            
        elif species == "electron":
            self.m = m_e
            self.q_sign = -1
            self.qm_ratio = (self.q_sign*e) / self.m
            self.filename = "electron_trajectory"
            
    def RungeKutta4(self, t, s, h):
        particle_escaped = False
        k1 = h * self.system(t, s)
        k2 = h * self.system(t + h/2, s + k1/2)
        k3 = h * self.system(t + h/2, s + k2/2)
        k4 = h * self.system(t + h, s + k3)
        s_new = s + (k1 + 2*k2 + 2*k3 + k4) / 6
        if np.sqrt(s_new[0]**2 + s_new[1]**2 + s_new[2]**2) > 5*Re:
            print("Particle escaped")
            particle_escaped = True
        return s_new, particle_escaped
    
    def Boris(self, t, s, h, Bfield):
        """
        Boris pusher method for advancing particle trajectories in a magnetic field,
        adapted to the provided notation.
        
        Parameters:
        t (float): Current time
        s (numpy.ndarray): Current state vector [x, y, z, vx, vy, vz]
        h (float): Time step
        
        Returns:
        numpy.ndarray: Updated state vector
        """
        particle_escaped = False
        # Extract position and velocity
        x_k = s[:3]  # Position: [x, y, z]
        v_k = s[3:]  # Velocity: [vx, vy, vz]
        
        # Calculate magnetic field at the particle's position        
        B_k = Bfield(x_k[0], x_k[1], x_k[2])

        # Assume no electric field (E_k = 0)
        E_k = np.zeros(3)
        
        # Precompute constants
        q_prime = h * self.qm_ratio/2  # q' = Δt * (q / 2m)
        h_vec = q_prime * B_k        # h = q' * B_k
        h_mag2 = np.dot(h_vec, h_vec)
        s_vec = 2 * h_vec / (1 + h_mag2)  # s = 2h / (1 + h^2)
        
        # Half-step velocity update due to E field
        v_minus = v_k + q_prime * E_k  # u = v_k-1/2 + q'E_k
        
        # Rotate velocity in the magnetic field
        v_prime = v_minus + np.cross(v_minus, h_vec)  # u' = u + (u × h)
        v_plus = v_minus + np.cross(v_prime, s_vec)   # u' = u' + (u' × s)
        
        # Full-step velocity update due to E field
        v_k_plus_half = v_plus + q_prime * E_k  # v_k+1/2 = u' + q'E_k
        
        # Position update
        x_k_plus_one = x_k + h * v_k_plus_half  # x_k+1 = x_k + Δt * v_k+1/2
        
        # Check for particle escape
        if np.sqrt(x_k_plus_one[0]**2 + x_k_plus_one[1]**2 + x_k_plus_one[2]**2) > 5 * Re:
            print("Particle escaped")
            particle_escaped = True
        
        # Return the updated state vector
        return np.concatenate((x_k_plus_one, v_k_plus_half)), particle_escaped

    def solve(self, s0, t_span, method, h, Bfield=None, dump=1):
        """
        Solve the system of differential equations using the specified method.
        
        Parameters:
            s0 (numpy array): Initial state [x, y, z, vx, vy, vz]
            t_span (list): Time span [t0, tf]
            method (str): "RK4", "Boris"
            h (float): Time step
            Bfield (function): Function that returns the magnetic field at a point (x, y, z), as an array [Bx, By, Bz]. Required for Boris method.
            dump (int): Number of steps to skip between saved states
            
        Returns:
            numpy array: Array of states [x, y, z, vx, vy, vz] at each time step.
        """
        t0, tf = t_span
        t = np.arange(t0, tf, h)
        s = s0
        S = []
        for i in range(len(t)):
            if dump == 1:
                S.append(s)
            elif i % dump == 0:
                S.append(s)
            if method == "RK4":
                s, particle_escaped = self.RungeKutta4(t[i], s, h)
            elif method == "Boris":
                s, particle_escaped = self.Boris(t[i], s, h, Bfield)
            if particle_escaped == True:
                break
        return np.array(S)
         
    def save_to_vtk_with_velocity(self, x, y, z, vx, vy, vz):
        """
        Save particle trajectory (x, y, z) and velocity magnitudes to a VTK file.
        
        Parameters:
            filename (str): Output VTK filename.
            x, y, z (numpy arrays): Position arrays of the particle trajectory.
            vx, vy, vz (numpy arrays): Velocity components of the particle.
        """
        # Create a vtkPoints object to store the trajectory points
        points = vtk.vtkPoints()
        for i in range(len(x)):
            points.InsertNextPoint(x[i], y[i], z[i])
        
        # Calculate speed (magnitude of velocity) at each point
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        
        # Create a vtkFloatArray to store the speed as point data
        speed_array = vtk.vtkFloatArray()
        speed_array.SetName("Speed")  # Name of the array (appears in ParaView)
        for s in speed:
            speed_array.InsertNextValue(s)
        
        # Create a vtkPolyLine to represent the trajectory as a connected line
        poly_line = vtk.vtkPolyLine()
        poly_line.GetPointIds().SetNumberOfIds(len(x))
        for i in range(len(x)):
            poly_line.GetPointIds().SetId(i, i)
        
        # Create a vtkCellArray to hold the polyline
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(poly_line)
        
        # Create a vtkPolyData object to hold the points and the polyline
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetLines(cells)
        
        # Add the speed data to the vtkPolyData object
        poly_data.GetPointData().AddArray(speed_array)
        poly_data.GetPointData().SetActiveScalars("Speed")
        
        # Write the vtkPolyData to a .vtk file
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(f"{self.filename}_velocity.vtk")
        writer.SetInputData(poly_data)
        writer.Write()
        print(f"Saved trajectory with velocity to {self.filename}_velocity.vtk")
    
    def save_to_html(self, x, y, z, filename):
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
    
    def plot(self, S, **kwargs):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # plot each 100 point
        ax.plot(S[:,0]/Re, S[:,1]/Re, S[:,2]/Re)
        ax.set_xlabel(r'X/R$_e$')
        ax.set_ylabel(r'Y/R$_e$')
        ax.set_zlabel(r'Z/R$_e$')
        # add planet
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='b')
        ax.set_title(kwargs.get('title', 'Particle trajectory'))
        ax.set_xlim(kwargs.get('xlim', [-5, 5]))
        ax.set_ylim(kwargs.get('ylim', [-5, 5]))
        ax.set_zlim(kwargs.get('zlim', [-5, 5]))
        plt.show()
        
def InitialVelocity(K, m):
    v = c * np.sqrt(1 - (1/((K/(m*c**2)) + 1))**2)
    return v

def InitialEnergy(v, m):
    K = m * c**2 * ((1/np.sqrt(1 - (v/c)**2)) - 1)
    return K

def Compute_B(x, y, z):
    """
    Compute the magnetic field at a point (x, y, z) in the Earth's magnetic field.
    
    Parameters:
        x, y, z (float): Coordinates in the Earth-centered frame.
        
    Returns:
        Bx, By, Bz (float): Magnetic field components at the point (x, y, z).
    """
    B0 = 3.07e-5  # T
    Re = 6378137  # Earth radius in meters
    B0_Re3 = B0 * Re**3
    r2 = x**2 + y**2 + z**2
    r5 = r2**2.5
    B_factor = -B0_Re3 / r5
    Bx = 3 * x * z * B_factor
    By = 3 * y * z * B_factor
    Bz = (2 * z**2 - x**2 - y**2) * B_factor
    return np.array([Bx, By, Bz])

def Compute_Vpar_Vperp(vx, vy, vz, Bx, By, Bz):
    """
    Compute the parallel and perpendicular components of the velocity with respect to the magnetic field.
    
    Parameters:
        vx, vy, vz (float): Velocity components in the Earth-centered frame.
        Bx, By, Bz (float): Magnetic field components in the Earth-centered frame.
        
    Returns:
        v_par, v_perp (float): Parallel and perpendicular components of the velocity.
    """
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
    v_dot_B = vx * Bx + vy * By + vz * Bz
    v_par = v_dot_B / B_mag
    v_perp = np.sqrt(v_mag**2 - v_par**2)
    return v_par, v_perp