import matplotlib.pyplot as plt
import numpy as np
import shapely as shape

from mesh.MESH2D import *
from cfdtd.CFDTD2D import *
from initial_pulse.INITIAL_PULSE2D import *

def test_visual_animation():
    node_list = [(25.5,10.5), (75.5,10.5), (75.5, 95.5), (25.5, 95.5)]
    initial_cell = (50, 50)
    mesh = Mesh(box_size=100.0, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell=initial_cell)
    # mesh.plotElectricFieldGrid()
    pulse = InitialPulse(mesh=mesh, initial_position=initial_cell, spread=10.0, pulse_type='Magnetic Gaussian')
    solver = CFDTD2D(mesh, initialPulse=pulse, cfl=1.0)

    # nsteps = int(100 / solver.dt)
    # solver.plotMagneticFieldAnimation(nsteps)
    
    solver.plotMagneticFieldFrame(13)

def test_rectangular_resonant_cavity44_Conformal():
    node_list = [(10.5,10.5), (95.5,10.5), (95.5, 95.5), (10.5, 95.5)]
    initial_cell = (50, 50)
    mesh = Mesh(box_size=100.0, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell=initial_cell)
    # mesh.plotElectricFieldGrid()
    pulse = InitialPulse(mesh=mesh, initial_position=initial_cell, spread=10.0, m=4, n=4, pulse_type='Rectangular Resonant Cavity 11')
    solver = CFDTD2D(mesh, initialPulse=pulse, cfl=0.4)
    nsteps = int(200 / solver.dt)

    # probeEx, probeEy, probeHz, probeTime = solver.run(nsteps)
    # node = (51, 51)
    # plt.plot(probeHz[node[0], node[1], :])
    # plt.title(f'Evolution in time of the magnetic field $H_z$ in the node {node}')
    # plt.xlabel('Time [ns]')
    # plt.ylabel('Magnetic field [A/m]')
    # plt.grid(True)
    # plt.show()
    
    # solver.plotMagneticFieldAnimation(nsteps)

    # solver.plotMagneticFieldFrame(14)

def test_rectangular_resonant_cavity44_NonConformal():
    node_list = [(10.5,10.5), (95.5,10.5), (95.5, 95.5), (10.5, 95.5)]
    initial_cell = (50, 50)
    mesh = Mesh(box_size=100.0, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell=initial_cell)
    # mesh.plotElectricFieldGrid()
    pulse = InitialPulse(mesh=mesh, initial_position=initial_cell, spread=10.0, m=4, n=4, pulse_type='Non Conformal Rectangular Resonant Cavity 11')
    solver = CFDTD2D(mesh, initialPulse=pulse, cfl=0.4, solver_type="Non Conformal")


    # nsteps = int(100 / solver.dt)
    # solver.plotMagneticFieldAnimation(nsteps)

    solver.plotMagneticFieldFrame(130)

def test_Hy_Gaussian():
    node_list = [(5.5,40.5), (94.5,40.5), (94.5, 60.5), (5.5, 60.5)]
    initial_cell = (50, 50)
    mesh = Mesh(box_size=100.0, dx=2.0, dy=2.0, external_nodes_list_PEC=node_list, initial_wave_cell=initial_cell)
    # mesh.plotElectricFieldGrid()
    pulse = InitialPulse(mesh=mesh, initial_position=initial_cell, spread=20.0, pulse_type='MagneticY Gaussian')
    solver = CFDTD2D(mesh, initialPulse=pulse, cfl=0.3)

    # nsteps = int(100 / solver.dt)
    # solver.plotMagneticFieldAnimation(nsteps)

    solver.plotMagneticFieldFrame(130)
    
    # for i in range(nsteps):
    #     plt.plot(mesh.gridHx, probeHz[:, 25, i], '.-', label='Electric Field')
    #     plt.ylim(-2.1, 2.1)
    #     plt.grid(which='both')
    #     plt.pause(0.01)  
    #     plt.cla()