import matplotlib.pyplot as plt
import numpy as np
import shapely as shape

from mesh.MESH2D import *
from cfdtd.CFDTD2D import *

def test_visual_animation():
    node_list = [(25.5,10.5), (75.5,10.5), (75.5, 95.5), (25.5, 95.5)]
    initial_cell = (50, 50)
    mesh = Mesh(box_size=100.0, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell=initial_cell)
    mesh.plotElectricFieldGrid()
    pulse = InitialPulse(initial_position=initial_cell, spread=5.0, pulse_type='Magnetic Gaussian')
    solver = CFDTD2D(mesh, initialPulse=pulse, cfl=0.9)
    nsteps = int(100 / solver.dt)
    probeEx, probeEy, probeHz, probeTime = solver.run(nsteps)

    solver.plotAnimation(nsteps)