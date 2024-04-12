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
    nsteps = 1000
    
    mesh = Mesh(box_size=200.0, pec_sheet_position=100.5, dx=1.0)
    pulse = InitialPulse(initial_time=40, initial_position=40, spread=10, pulse_type="Gaussian")
    solver = CFDTD1D(mesh, pulse, boundary_type="periodic", cfl=0.8)
    probeE, probeH = solver.run(nsteps)

    assert np.allclose(probeE[mesh.getPECIndexPosition(), :], np.zeros(nsteps))