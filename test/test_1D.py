import numpy as np
from math import exp
from matplotlib import pyplot as plt

from cfdtd.CFDTD1D import *

Cells = 120
PEC_sheet = 119.5
cfl = 0.5

def test_pec_pulse():
    mesh = Mesh(10.0, 0.1)
    solver = CFDTD1D(mesh, 1.0, 40, 10)
    probeE, probeH = solver.run(1000)

    for n in range(probeE.shape[1]):
        plt.plot(mesh.vx, probeE[:,n])
        plt.pause(0.01)
        plt.grid(which='both')
        plt.ylim(-1.1, 1.1)
        plt.cla()


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