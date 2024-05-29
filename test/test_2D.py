import matplotlib.pyplot as plt
import numpy as np
import shapely as shape

from mesh.MESH2D import *
from cfdtd.CFDTD2D import *

def test_visual_animation():
    node_list = [(85.5,10.5), (95.5,10.5), (95.5, 95.5), (85.5, 95.5)]
    initial_cell = (50, 50)
    mesh = Mesh(box_size=100.0, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell=initial_cell)
    # mesh.plotElectricFieldGrid()
    solver = CFDTD2D(mesh, initial_position=initial_cell, spread=1.0, cfl=0.99)
    nsteps = int(100 / solver.dt)
    probeEx, probeEy, probeHz, probeTime = solver.run(nsteps)

    solver.plotAnimation(nsteps)