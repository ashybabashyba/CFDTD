import matplotlib.pyplot as plt
import numpy as np
import shapely as shape

from mesh.MESH2D import *
from cfdtd.CFDTD2D import *
from initial_pulse.INITIAL_PULSE2D import *

c0 = 3*(10**8)

def test_visual_animation():
    node_list = [(25.5,10.5), (75.5,10.5), (75.5, 95.5), (25.5, 95.5)]
    initial_cell = (50, 50)
    mesh = Mesh(box_size=100.0, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell=initial_cell)
    # mesh.plotElectricFieldGrid()
    pulse = InitialPulse(mesh=mesh, initial_position=initial_cell, spread=10.0, pulse_type='Magnetic Gaussian')
    solver = CFDTD2D(mesh, initialPulse=pulse, cfl=1.0)

    # nsteps = int(100 / solver.dt)
    # solver.plotMagneticFieldAnimation(nsteps)
    
    solver.plotMagneticFieldFrame(433)

def test_rectangular_resonant_cavity44_Conformal():
    node_list = [(10.5,10.5), (95.5,10.5), (95.5, 95.5), (10.5, 95.5)]
    initial_cell = (50, 50)
    mesh = Mesh(box_size=100.0, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell=initial_cell)
    # mesh.plotElectricFieldGrid()
    pulse = InitialPulse(mesh=mesh, initial_position=initial_cell, spread=10.0, m=4, n=4, pulse_type='Rectangular Resonant Cavity')
    solver = CFDTD2D(mesh, initialPulse=pulse, cfl=0.4)
    nsteps = int(200 / solver.dt)

    # probeEx, probeEy, probeHz, probeTime = solver.run(nsteps)
    # node = (51, 51)

    # t = np.linspace(0, 700, num=10000)
    # a = np.abs(mesh.nodesList[0][0] - mesh.nodesList[1][0])
    # b = np.abs(mesh.nodesList[0][1] - mesh.nodesList[3][1])
    # x0 = mesh.nodesList[0][0]
    # y0 = mesh.nodesList[0][1]
    # frec = c0*np.sqrt( (pulse.m/a )**2 + (pulse.n/b )**2)/2/1e9
    # H_teo = np.cos(pulse.m*np.pi*(mesh.gridHx[node[0]] - x0)/a)*np.cos(pulse.n*np.pi*(mesh.gridHy[node[1]] - y0)/b)*np.cos(frec*t*2*np.pi)

    # fig, ax = plt.subplots()
    # ax.plot(probeTime, probeHz[node[0], node[1], :], linestyle='--', label='Numerical magnetic field')
    # ax.plot(t, H_teo, label='Theoretical magnetic field')
    # ax.set_title(f'Evolution in time of the magnetic field $H_z$ in the node {node}')
    # ax.set_xlabel('Time [ns]')
    # ax.set_ylabel('Magnetic field [A/m]')
    # ax.set_xlim(0, 300)
    # ax.grid(True)
    # ax.legend(loc='lower right')

    # axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    # axins.plot(probeTime, probeHz[node[0], node[1], :], linestyle='--')
    # axins.plot(t, H_teo)
    # axins.set_xlim(125, 150)
    # axins.set_ylim(-1.0, -0.75)
    # ax.indicate_inset_zoom(axins)

    # plt.show()
    
    # solver.plotMagneticFieldAnimation(nsteps)

    solver.plotMagneticFieldFrame(99)
    

def test_rectangular_resonant_cavity44_NonConformal():
    node_list = [(10.5,10.5), (95.5,10.5), (95.5, 95.5), (10.5, 95.5)]
    initial_cell = (50, 50)
    mesh = Mesh(box_size=100.0, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell=initial_cell)
    # mesh.plotElectricFieldGrid()
    pulse = InitialPulse(mesh=mesh, initial_position=initial_cell, spread=10.0, m=4, n=4, pulse_type='Non Conformal Rectangular Resonant Cavity')
    solver = CFDTD2D(mesh, initialPulse=pulse, cfl=0.4, solver_type="Non Conformal")


    nsteps = int(100 / solver.dt)
    solver.plotMagneticFieldAnimation(nsteps)

    # solver.plotMagneticFieldFrame(130)

def test_Hy_Gaussian():
    node_list = [(5.5,40.5), (94.5,40.5), (94.5, 60.5), (5.5, 60.5)]
    initial_cell = (50, 50)
    mesh = Mesh(box_size=100.0, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell=initial_cell)
    # mesh.plotElectricFieldGrid()
    pulse = InitialPulse(mesh=mesh, initial_position=initial_cell, spread=20.0, pulse_type='MagneticY Gaussian')
    solver = CFDTD2D(mesh, initialPulse=pulse, cfl=0.3)

    # nsteps = int(100 / solver.dt)
    # solver.plotMagneticFieldAnimation(nsteps)

    frame = 140 
    solver.plotMagneticFieldFrame(frame)

    nsteps = int(frame*c0/solver.dt/1e9)
    probeEx, probeEy, probeHz, probeTime = solver.run(nsteps+1)

    plt.plot(mesh.gridHx, probeHz[:, 50, nsteps], '.-')
    plt.ylim(-2.1, 2.1)
    plt.title(f'Time: {probeTime[nsteps]:.2f} [ns]')
    plt.xlabel('x coordinate [m]')
    plt.ylabel('$H_z$ [A/m]')
    plt.grid(which='both')
    plt.show()
    
    # for i in range(nsteps):
    #     plt.plot(mesh.gridHx, probeHz[:, 25, i], '.-', label='Electric Field')
    #     plt.ylim(-2.1, 2.1)
    #     plt.grid(which='both')
    #     plt.pause(0.01)  
    #     plt.cla()

def test_Hy_Illumination():
    node_list = [(5.5,40.5), (94.5,40.5), (94.5, 60.5), (5.5, 60.5)]
    initial_cell = (50, 50)
    mesh = Mesh(box_size=100.0, dx=1.0, dy=1.0, external_nodes_list_PEC=node_list, initial_wave_cell=initial_cell)
    # mesh.plotElectricFieldGrid()
    xmin, xmax, ymin, ymax = mesh.getMinMaxIndexInsideNonConformalCells()

    pulse = InitialPulse(mesh=mesh, initial_position=initial_cell, spread=20.0)
    solver = CFDTD2D(mesh, initialPulse=pulse, cfl=0.3)

    for j in range(ymin-1, ymax+2):
        solver.addSource(pulse.illuminationGaussianHy(locationX=xmin-1, locationY=j, center=10, amplitude=0.5, spread=1))


    nsteps = int(100 / solver.dt)
    solver.plotMagneticFieldAnimation(nsteps)

    # frame = 140 
    # solver.plotMagneticFieldFrame(frame)

    # nsteps = int(frame*c0/solver.dt/1e9)
    # probeEx, probeEy, probeHz, probeTime = solver.run(nsteps+1)

    # plt.plot(mesh.gridHx, probeHz[:, 50, nsteps], '.-')
    # plt.ylim(-2.1, 2.1)
    # plt.title(f'Time: {probeTime[nsteps]:.2f} [ns]')
    # plt.xlabel('x coordinate [m]')
    # plt.ylabel('$H_z$ [A/m]')
    # plt.grid(which='both')
    # plt.show()