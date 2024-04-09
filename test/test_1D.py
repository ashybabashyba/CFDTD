import numpy as np
from math import exp
from matplotlib import pyplot as plt

from cfdtd.CFDTD1D import *

Cells = 120
PEC_sheet = 119.5
cfl = 0.5

def test_pec_pulse():
    mesh = Mesh(100.0, 1.0)
    pulse = InitialPulse(40, 10)
    solver = CFDTD1D(mesh, pulse, "Gaussian", 1.0)
    probeE, probeH = solver.run(500)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    line1, = ax1.plot(probeE[:,0], color='k', linewidth=1)
    line2, = ax2.plot(probeH[:,0], color='k', linewidth=1)
    ax1.axvline(x=100.5, color='r', linestyle='--', label='kp')
    ax2.axvline(x=100.5, color='r', linestyle='--', label='kp')

    # Function to initialize the plot
    def init():
        ax1.set_ylabel('E$_x$', fontsize='14')
        ax1.set_xlim(0, 100)
        ax1.set_ylim(-2.2, 2.2)

        ax2.set_ylabel('H$_y$', fontsize='14')
        ax2.set_xlabel('FDTD cells')
        ax2.set_xlim(0, 100)
        ax2.set_ylim(-2.2, 2.2)
        plt.subplots_adjust(bottom=0.2, hspace=0.45)
        return line1, line2


    def animate(i):
        line1.set_ydata(probeE[:,i])
        line2.set_ydata(probeH[:,i])
        return line1, line2


    # Create the animation
    ani = FuncAnimation(fig, animate, frames=1001, init_func=init, blit=True, interval=10, repeat=False)

    # Show the animation
    plt.show()

def test_mesh_comparative():
    conformal_mesh = cfdtd.SpatialMesh()
    non_conformal_mesh = fdtd.SpatialMesh()

    if Cells == PEC_sheet:
        assert np.array_equal(conformal_mesh, non_conformal_mesh)
    else:
        assert (not np.array_equal(conformal_mesh, non_conformal_mesh))


def test_field_comparative_with_equal_mesh():
    conformal_fields = cfdtd.run()
    non_conformal_fields = fdtd.FDTDLoop()

    if (Cells == PEC_sheet) and (cfdtd.CourantConditionNumber() <= 1):
        assert np.array_equal(conformal_fields[0], non_conformal_fields[0])
        assert np.array_equal(conformal_fields[1], non_conformal_fields[1])