import numpy as np
from math import exp
from matplotlib import pyplot as plt

from cfdtd.CFDTD1D import *

c0 = 3*(10**8)


def test_visual_pec_pulse():
    mesh = Mesh(box_size=200.0, pec_sheet_position=150.5, dx=1.0)
    pulse = InitialPulse(initial_time=40, initial_position=80, spread=10, pulse_type="Gaussian")
    solver = CFDTD1D(mesh, pulse, boundary_type="pec", cfl=1.0)
    probeE, probeH = solver.run(100)

    # plt.hlines(1,0,20)  
    # plt.xlim(0,21)
    # plt.ylim(0.5,1.5)

    # y = np.ones(np.shape(mesh.xE))   
    # plt.plot(mesh.xE,y,'|', ms=20)
    # plt.scatter(mesh.getPECSheetPosition(), 1, color='red', s=500, zorder=10, marker='|')

    # node1_x = mesh.xE[mesh.getPECIndexPosition()-1]  
    # node2_x = mesh.getPECSheetPosition()  
    # node3_x = mesh.xE[mesh.getPECIndexPosition()+1]
    # delta_y = 1.0  

    # plt.annotate(
    #     '$\Delta x_l$',
    #     xy=((node1_x + node2_x) / 2, delta_y),
    #     xytext=((node1_x + node2_x) / 2, delta_y + 0.1),
    #     arrowprops=dict(arrowstyle='->', lw=.5),
    #     ha='center'
    # )

    # plt.annotate(
    #     '$\Delta x_r$',
    #     xy=((node3_x + node2_x) / 2, delta_y),
    #     xytext=((node3_x + node2_x) / 2, delta_y - 0.1),
    #     arrowprops=dict(arrowstyle='->', lw=.5),
    #     ha='center'
    # )
    # plt.title('Example of an electric field mesh')
    # plt.axis('off')
    # plt.show()

    for n in range(probeE.shape[1]):
        if n % 1 == 0:  # Skipping rate 
            plt.plot(mesh.xE, probeE[:,n], '.-', label='Electric Field')
            #plt.plot(mesh.xH, probeH[:,n], '.-', label='Magnetic Field')
            plt.axvline(x=mesh.getPECSheetPosition(), color='r', linestyle='--', label='PEC sheet position')
            plt.legend()
            plt.ylim(-2.1, 2.1)
            plt.xlabel('x position [m]')
            plt.ylabel('$E_x$ [V/m]')
            plt.title(f'Time: {n*solver.dt*1e9/c0:.2f} [ns]')
            plt.grid(which='both')
            plt.pause(0.01)  
            plt.cla()


def test_ElectricField_on_PEC():
    final_time = 1000
    
    mesh = Mesh(box_size=200.0, pec_sheet_position=100.5, dx=1.0)
    pulse = InitialPulse(initial_time=40, initial_position=40, spread=10, pulse_type="Gaussian")
    solver = CFDTD1D(mesh, pulse, boundary_type="periodic", cfl=0.8)
    probeE, probeH = solver.run(final_time)

    assert np.allclose(probeE[mesh.getPECIndexPosition(), :], np.zeros(int(final_time/solver.dt)))

def test_ElectricField_delay():
    courantNumber = 0.5
    final_time = 600 
    pulse = InitialPulse(initial_time=40, initial_position=100, spread=10, pulse_type="Gaussian")

    conformalMesh = Mesh(box_size=200.0, pec_sheet_position=199.5, dx=1.0)
    nonConformalMesh = Mesh(box_size=200.0, pec_sheet_position=200, dx=1.0)

    conformalSolver = CFDTD1D(conformalMesh, pulse, boundary_type="pec", cfl=courantNumber)
    nonConformalSolver = CFDTD1D(nonConformalMesh, pulse, boundary_type="pec", cfl=courantNumber)

    conformalE, conformalH = conformalSolver.run(final_time)
    nonConformalE, nonConformalH = nonConformalSolver.run(final_time)

    # plt.plot(conformalMesh.xE, conformalE[:,conformalE.shape[1]-1], '.-', label='Conformal Electric Field')
    # plt.plot(nonConformalMesh.xE, nonConformalE[:,conformalE.shape[1]-1], '.-', label='Non Conformal Electric Field')
    # plt.axvline(x=nonConformalMesh.getPECSheetPosition(), color='r', linestyle='--', label='PEC sheet position')
    # plt.legend()
    # plt.ylim(-2.1, 2.1)
    # plt.title(f'Time: {final_time*conformalSolver.dt*1e9/c0:.2f} [ns]')
    # plt.xlabel('x position [m]')
    # plt.ylabel('$E_x$ [V/m]')
    # plt.grid(which='both')
    # plt.show()

    # plt.figure(figsize=(8, 6))
    # error_norm1 = np.abs(conformalE[:-1, final_time] - nonConformalE[:, final_time])
    # plt.scatter(range(len(error_norm1)), error_norm1, marker='o', color='b')
    # plt.title(f'Absolute difference between non conformal solution \n and conformal solution at {final_time*conformalSolver.dt*1e9/c0:.2f} [ns]')
    # plt.ylim(-0.05, 0.5)
    # plt.xlabel('x position [m]')
    # plt.ylabel('$E_x$ [V/m]')
    # plt.grid(True)
    # plt.show()

    assert np.allclose(conformalE[:-1, final_time-2], nonConformalE[:, final_time], atol=5.e-4)
    assert np.allclose(conformalH[:-1, final_time-2], nonConformalH[:, final_time], atol=5.e-4)

