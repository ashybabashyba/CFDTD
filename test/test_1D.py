import numpy as np
from math import exp
from matplotlib import pyplot as plt

from cfdtd.CFDTD1D import *


def test_pec_pulse():
    mesh = Mesh(box_size=200.0, pec_sheet_position=100.5, dx=1.0)
    pulse = InitialPulse(initial_time=40, initial_position=40, spread=10, pulse_type="Gaussian")
    solver = CFDTD1D(mesh, pulse, boundary_type="periodic", cfl=0.5)
    probeE, probeH = solver.run(1000)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    line1, = ax1.plot(probeE[:,0], color='k', linewidth=1)
    line2, = ax2.plot(probeH[:,0], color='k', linewidth=1)
    ax1.axvline(x=mesh.getPECSheetPosition(), color='r', linestyle='--', label='kp')
    ax2.axvline(x=mesh.getPECSheetPosition(), color='r', linestyle='--', label='kp')
    ax1.set_title(mesh.getPECIndexPosition())

    # Function to initialize the plot
    def init():
        ax1.set_ylabel('E$_x$', fontsize='14')
        ax1.set_xlim(0, mesh.getLength())
        ax1.set_ylim(-2.2, 2.2)

        ax2.set_ylabel('H$_y$', fontsize='14')
        ax2.set_xlabel('FDTD cells')
        ax2.set_xlim(0, mesh.getLength())
        ax2.set_ylim(-2.2, 2.2)
        plt.subplots_adjust(bottom=0.2, hspace=0.45)
        return line1, line2


    def animate(i):
        line1.set_ydata(probeE[:,i])
        line2.set_ydata(probeH[:,i])
        return line1, line2


    # Create the animation
    ani = FuncAnimation(fig, animate, frames=1001, init_func=init, blit=True, interval=3.5, repeat=False)

    # Show the animation
    plt.show()

def test_ElectricField_on_PEC():
    nsteps = 1000
    
    mesh = Mesh(box_size=200.0, pec_sheet_position=100.5, dx=1.0)
    pulse = InitialPulse(initial_time=40, initial_position=40, spread=10, pulse_type="Gaussian")
    solver = CFDTD1D(mesh, pulse, boundary_type="periodic", cfl=0.8)
    probeE, probeH = solver.run(nsteps)

    assert np.allclose(probeE[mesh.getPECIndexPosition(), :], np.zeros(nsteps))