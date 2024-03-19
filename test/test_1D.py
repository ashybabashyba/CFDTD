import numpy as np
from math import exp
from matplotlib import pyplot as plt

from fdtd.FDTD1D import FDTD1D_Class
from cfdtd.CFDTD1D import CFDTD1D_Class


Cells = 120
PEC_sheet = 119.5
cfl = 0.5

fdtd = FDTD1D_Class(ke=Cells, cfl=cfl, t0=40, spread=10, nsteps=1000)
cfdtd = CFDTD1D_Class(ke=Cells, kp=PEC_sheet, cfl=cfl, t0=40, spread=10, nsteps=1000)


def test_mesh_comparative():
    conformal_mesh = cfdtd.SpatialMesh()
    non_conformal_mesh = fdtd.SpatialMesh()

    if Cells == PEC_sheet:
        assert np.array_equal(conformal_mesh, non_conformal_mesh)
    else:
        assert (not np.array_equal(conformal_mesh, non_conformal_mesh))


def test_field_comparative_with_equal_mesh():
    conformal_fields = cfdtd.CFDTDLoop()
    non_conformal_fields = fdtd.FDTDLoop()

    if (Cells == PEC_sheet) and (cfdtd.CourantConditionNumber() <= 1):
        assert np.array_equal(conformal_fields[0], non_conformal_fields[0])
        assert np.array_equal(conformal_fields[1], non_conformal_fields[1])