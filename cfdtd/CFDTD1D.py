import numpy as np
from math import exp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

class Mesh():
    def __init__(self, box_size, pec_sheet_position, dx):
        self.dx = dx
        self.kp = pec_sheet_position 
        V1 = np.linspace(0, box_size, int(1 + box_size/dx))
        V2 = np.array([self.kp])
        if any(np.isclose(self.kp, x) for x in V1):
            self.vx = V1
            self.leftLeftover = 1.0
            self.rightLeftover = 1.0
        else:
            self.vx = np.sort(np.concatenate((V1, V2)))
            self.leftLeftover = self.dx/(self.kp - self.vx[np.searchsorted(self.vx, self.kp)-1])
            self.rightLeftover =  self.leftLeftover/(self.leftLeftover-1)

    def getDX(self):
        return self.dx
    
    def getLength(self):
        return self.vx[-1]

    def getLeftLeftover(self):
        return self.leftLeftover
    
    def getRightLeftover(self):
        return self.rightLeftover

    def getPECSheetPosition(self):
        return self.kp    
    
    def numberOfCells(self):
        return len(self.vx) - 1
    
    def numberOfNodes(self):
        return len(self.vx)
    
    def getPECIndexPosition(self):
        return np.searchsorted(self.vx, self.kp)

    def getSpatialDiscretization(self):
        return self.vx

class InitialPulse():
    def __init__(self, initial_time, initial_position, spread, pulse_type):
        self.t0 = initial_time
        self.x0 = initial_position
        self.spread = spread
        self.type = pulse_type

    def getT0(self):
        return self.t0
    
    def getSpread(self):
        return self.spread
    
    def initialPosition(self):
        return self.x0

    def pulse(self, dx, dt, step):
        if self.type == "Gaussian":
            E0 = dt*exp(-0.5 * ((self.t0 - dt*step) / self.spread) ** 2)
            H0 = dt*exp(-0.5 * ((self.t0 - dt/2 - dx/2 - dt*step) / self.spread) ** 2)
            return E0, H0
        else:
            raise ValueError("Pulse not defined")

class CFDTD1D():
    def __init__(self, mesh, initialPulse, boundary_type, cfl):
        
        self.mesh = mesh
        self.pulse = initialPulse
        
        self.cfl = cfl
        self.boundary = boundary_type

        self.t0 = self.pulse.getT0()
        self.spread = self.pulse.getSpread()

        self.kp = self.mesh.getPECSheetPosition()
        self.ke = self.mesh.getLength()
        self.kc = self.pulse.initialPosition()

        self.dx = self.mesh.getDX()
        self.dt = self.dx * self.cfl
        self.cb = self.dt/self.dx
        self.leftLeftover = self.mesh.getLeftLeftover()
        self.rightLeftover = self.mesh.getRightLeftover()
    
    def buildFields(self):
        ex = np.zeros(self.mesh.numberOfNodes())
        hy = np.zeros(self.mesh.numberOfNodes()-1)

        return ex, hy
    
    def buildFieldsInAllTimeSteps(self, nsteps):
        ex, hy = self.buildFields()
        probeE = np.zeros((len(ex), nsteps))
        probeH = np.zeros((len(hy), nsteps))

        return probeE, probeH
    
    def run(self, nsteps):
        ex, hy = self.buildFields()
        probeE, probeH = self.buildFieldsInAllTimeSteps(nsteps)
        IndexPEC = self.mesh.getPECIndexPosition()

        for time_step in range(nsteps):
            ex[1:IndexPEC-1] += self.cb*(hy[0:IndexPEC-2] - hy[1:IndexPEC-1])
            ex[IndexPEC-1] += self.leftLeftover * self.cb * (hy[IndexPEC - 2] - hy[IndexPEC-1])
            if IndexPEC+1 < self.mesh.numberOfNodes()-1:
                ex[IndexPEC+1] += self.rightLeftover * self.cb * (hy[IndexPEC] - hy[IndexPEC+1])
                ex[IndexPEC+2:-1] += self.cb*(hy[IndexPEC+1:-1] - hy[IndexPEC+2:])

            Wave = self.pulse.pulse(self.dx, self.dt, time_step)
            ex[self.kc] += Wave[0]
            hy[self.kc] += Wave[1]

            hy[:IndexPEC-1] += self.cb*(ex[:IndexPEC-1] - ex[1:IndexPEC])
            hy[IndexPEC-1] += self.leftLeftover * self.cb * ex[IndexPEC-1]
            if IndexPEC + 1 < self.mesh.numberOfCells()-1:
                hy[IndexPEC] -= self.rightLeftover * self.cb * ex[IndexPEC+1]
                hy[IndexPEC+1:] += self.cb * (ex[IndexPEC+1:-1] - ex[IndexPEC+2:])

            if self.boundary == "pec":
                ex[0] = 0.0
                ex[-1] = 0.0
            elif self.boundary == "periodic":
                ex[0] += - self.cb * (hy[0] - hy[-1])
                ex[-1] = ex[0]
            else:
                raise ValueError("Boundary not defined")

            probeE[:,time_step]=ex[:]
            probeH[:,time_step]=hy[:]
            
        return probeE, probeH