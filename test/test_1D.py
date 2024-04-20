import numpy as np
from math import exp
from matplotlib import pyplot as plt

from cfdtd.CFDTD1D import *


def test_visual_pec_pulse():
    mesh = Mesh(box_size=200.0, pec_sheet_position=199.5, dx=1.0)
    pulse = InitialPulse(initial_time=40, initial_position=50, spread=10, pulse_type="Gaussian")
    solver = CFDTD1D(mesh, pulse, boundary_type="periodic", cfl=0.5)
    probeE, probeH = solver.run(1000)

    for n in range(probeE.shape[1]):
        if n % 15 == 0:  # Skipping rate 
            plt.plot(mesh.xE, probeE[:,n], '.-', label='Electric Field')
            plt.plot(mesh.xH, probeH[:,n], '.-', label='Magnetic Field')
            plt.axvline(x=mesh.getPECSheetPosition(), color='r', linestyle='--', label='PEC sheet position')
            plt.legend()
            plt.ylim(-2.1, 2.1)
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
    final_time = 800 
    pulse = InitialPulse(initial_time=40, initial_position=100, spread=10, pulse_type="Gaussian")

    conformalMesh = Mesh(box_size=200.0, pec_sheet_position=200, dx=1.0)
    nonConformalMesh = Mesh(box_size=200.0, pec_sheet_position=199.5, dx=1.0)

    conformalSolver = CFDTD1D(conformalMesh, pulse, boundary_type="pec", cfl=courantNumber)
    nonConformalSolver = CFDTD1D(nonConformalMesh, pulse, boundary_type="pec", cfl=courantNumber)

    conformalE, conformalH = conformalSolver.run(final_time)
    nonConformalE, nonConformalH = nonConformalSolver.run(final_time)

    # plt.plot(conformalMesh.xE, conformalE[:,conformalE.shape[1]-1], '.-', label='Conformal Electric Field')
    # plt.plot(nonConformalMesh.xE, nonConformalE[:,conformalE.shape[1]-1], '.-', label='Non Conformal Electric Field')
    # plt.axvline(x=nonConformalMesh.getPECSheetPosition(), color='r', linestyle='--', label='PEC sheet position')
    # plt.legend()
    # plt.ylim(-2.1, 2.1)
    # plt.grid(which='both')
    # plt.show()

    for n in range(120):
        assert np.allclose(conformalE[:-1, n], nonConformalE[:-2, n], atol=1.e-1)
        assert np.allclose(conformalH[:-1, n], nonConformalH[:-2, n], atol=1.e-1)

    for n in range(150,500):
        assert np.allclose(conformalE[:-1, n], nonConformalE[:-2, n-1], atol=1.e-1)
        assert np.allclose(conformalH[:-1, n], nonConformalH[:-2, n-1], atol=1.e-1)
