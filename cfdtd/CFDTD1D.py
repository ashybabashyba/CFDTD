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
            self.leftover = 1.0
        else:
            self.vx = np.sort(np.concatenate((V1, V2)))
            self.leftover = self.dx/(self.kp - self.vx[np.searchsorted(self.vx, self.kp)-1])

    def getDX(self):
        return self.dx
    
    def getLength(self):
        return self.vx[-1]

    def getLeftover(self):
        return self.leftover

    def getPECSheetPosition(self):
        return self.kp    
    
    def numberOfCells(self):
        return len(self.vx) - 1
    
    def numberOfNodes(self):
        return len(self.vx)

class InitialPulse():
    def __init__(self, initial_time, spread):
        self.t0 = initial_time
        self.spread = spread

    def getT0(self):
        return self.t0
    
    def getSpread(self):
        return self.spread
    
    def initialPosition(self, box_size):
        return int(box_size / 5)

    def gaussianPulse(self, dx, dt, step):
        E0 = dt*exp(-0.5 * ((self.t0 - dt*step) / self.spread) ** 2)
        H0 = dt*exp(-0.5 * ((self.t0 - dt/2 - dx/2 - dt*step) / self.spread) ** 2)
        return E0, H0

class CFDTD1D():
    def __init__(self, mesh, initialPulse, type_of_Pulse, cfl):
        
        self.mesh = mesh
        self.pulse = initialPulse
        self.type_of_pulse = type_of_Pulse
        
        self.cfl = cfl
        self.t0 = self.pulse.getT0()
        self.spread = self.pulse.getSpread()

        self.kp = self.mesh.getPECSheetPosition()
        self.ke = self.mesh.getLength()
        self.kc = self.pulse.initialPosition(self.ke)

        self.dx = self.mesh.getDX()
        self.dt = self.dx * self.cfl
        self.cb = self.dt/self.dx
        self.leftover = self.mesh.getLeftover()
    
    def buildFields(self):
        ex = np.zeros(self.mesh.numberOfNodes())
        hy = np.zeros(self.mesh.numberOfNodes())

        return ex, hy
    
    def buildFieldsInAllTimeSteps(self, nsteps):
        ex, hy = self.buildFields()
        probeE = np.zeros((len(ex), nsteps))
        probeH = np.zeros((len(hy), nsteps))

        return probeE, probeH
    
    def run(self, nsteps):
        ex, hy = self.buildFields()
        probeE, probeH = self.buildFieldsInAllTimeSteps(nsteps)

        for time_step in range(nsteps):
            if self.type_of_pulse == "Gaussian":
                Wave = self.pulse.gaussianPulse(self.dx, self.dt, time_step)
            else:
                raise ValueError("Pulse not defined")

            for k in range(1, self.mesh.numberOfNodes()):
                if k<np.floor(self.kp):
                    ex[k] = ex[k] + self.cb * (hy[k - 1] - hy[k])
                elif k==np.floor(self.kp):
                    ex[k] = ex[k] + self.leftover * self.cb * (hy[k - 1] - hy[k])

            ex[self.kc] += Wave[0]
            hy[self.kc] += Wave[1]

            for k in range(self.mesh.numberOfNodes()):
                if k==np.floor(self.kp):
                    hy[k] = hy[k] + self.leftover * self.cb * (ex[k])
                elif k<np.floor(self.kp):
                    hy[k] = hy[k] + self.cb * (ex[k] - ex[k + 1])

            probeE[:,time_step]=ex[:]
            probeH[:,time_step]=hy[:]
            
        return probeE, probeH

