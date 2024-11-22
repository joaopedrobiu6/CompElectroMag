import meep as mp
import numpy as np
import matplotlib.pyplot as plt

cell = mp.Vector3(20, 20, 20)
geometry = [
    mp.Block(
        mp.Vector3(16, 16, 16),
        center=mp.Vector3(0, 0, 0),
        material=mp.Medium(epsilon=12),
    ),
]
resolution = 10

sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    resolution=resolution,
)

sim.plot3D()