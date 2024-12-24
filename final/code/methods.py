import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, m_e, c, m_p
import vtk
import plotly.graph_objects as go


Re = 6378137  # Earth radius in meters
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

    
    def solve(self, s0, t_span, method, h):
        t0, tf = t_span
        t = np.arange(t0, tf, h)
        s = s0
        S = []
        for i in range(len(t)):
            if i % 4000 == 0:
                # print(f"Time: {t[i]:.2f}")
                S.append(s)
            s, particle_escaped = self.RungeKutta4(t[i], s, h)
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
    
    def save_to_html(self, x, y, z):
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
        fig.write_html(f"{self.filename}.html")
    
    def plot(self, S):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # plot each 100 point
        ax.plot(S[:,0], S[:,1], S[:,2])
        ax.set_xlabel(r'X/R$_e$')
        ax.set_ylabel(r'Y/R$_e$')
        ax.set_zlabel(r'Z/R$_e$')
        # ax.set_xlim(-5, 5)
        # ax.set_ylim(-5, 5)
        # ax.set_zlim(-5, 5)
        plt.show()
        
