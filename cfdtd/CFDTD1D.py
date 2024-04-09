import numpy as np
from math import exp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

class Mesh():
    def __init__(self, box_size, dx):
        kpi = np.floor(box_size)
                
        V1 = np.linspace(0, kpi, int(1 + kpi/dx))
        V2 = np.array([box_size])
        if np.isclose(kpi, box_size):
            self.vx = V1
        else:
            self.vx = np.concatenate((V1, V2))

    def getDX(self):
        return self.vx[1] - self.vx[0]
    
    def getLength(self):
        return self.vx[-1]
    
    def getLeftover(self):
        kp = self.getLength()
        if kp==np.floor(kp):
            return 1.0
        else:
            return 1.0 /(kp-np.floor(kp))
    
    def numberOfCells(self):
        return len(self.vx) - 1

class InitialPulse():
    def __init__(self, initial_time, spread):
        self.t0 = initial_time
        self.spread = spread

    def getT0(self):
        return self.t0
    
    def getSpread(self):
        return self.spread

    def gaussianPulse(self, dx, dt, step):
        E0 = dt*exp(-0.5 * ((self.t0 - dt*step) / self.spread) ** 2)
        H0 = dt*exp(-0.5 * ((self.t0 - dt/2 - dx/2 - dt*step) / self.spread) ** 2)
        return E0, H0

class CFDTD1D():
    def __init__(self, mesh, initialPulse, cfl):
        
        self.mesh = mesh
        self.pulse = initialPulse
        
        self.cfl = cfl
        self.t0 = self.pulse.getT0()
        self.spread = self.pulse.getSpread()

        self.dx = self.mesh.getDX()
        self.dt = self.dx * self.cfl
        self.cb = self.dt/self.dx
    
    def buildFields(self):
        ex = np.zeros(self.mesh.numberOfCells()+1)
        hy = np.zeros(self.mesh.numberOfCells()+1)

        return ex, hy
    
    def buildFieldsInAllTimeSteps(self, nsteps):
        ex, hy = self.buildFields()
        probeE = np.zeros((len(ex), nsteps))
        probeH = np.zeros((len(hy), nsteps))

        return probeE, probeH
    
    def run(self, nsteps):
        ex, hy = self.buildFields()
        probeE, probeH = self.buildFieldsInAllTimeSteps(nsteps)
        
        kp = self.mesh.getLength()
        kc = int(kp / 5)
        self.ke = int(kp)

        leftover = self.mesh.getLeftover()

        for time_step in range(nsteps):

            for k in range(1, self.mesh.numberOfCells()):
                if k<np.floor(kp):
                    ex[k] = ex[k] + self.cb * (hy[k - 1] - hy[k])
                elif k==np.floor(kp):
                    ex[k] = ex[k] + leftover * self.cb * (hy[k - 1] - hy[k])

            Wave = self.pulse.gaussianPulse(self.dx, self.dt, time_step)
            ex[kc] += Wave[0]
            hy[kc] += Wave[1]
 

            for k in range(self.ke - 1):
                if k==np.floor(kp):
                    hy[k] = hy[k] + leftover * self.cb * (ex[k])
                elif k<np.floor(kp):
                    hy[k] = hy[k] + self.cb * (ex[k] - ex[k + 1])

            probeE[:,time_step]=ex[:]
            probeH[:,time_step]=hy[:]
            
        return probeE, probeH

