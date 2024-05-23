import numpy as np
import math
from matplotlib import pyplot as plt
import matplotlib.animation as animation

from mesh.MESH2D import Mesh

class CFDTD2D():
    def __init__(self, mesh, initial_position, spread, cfl):
        self.mesh = mesh
        self.center = initial_position
        self.cfl = cfl
        self.spread = spread

        self.dx = self.mesh.dx
        self.dy = self.mesh.dy
        self.dt = cfl*math.sqrt(self.dx**2 + self.dy**2)/2

    def buildFields(self):
        Ex = np.zeros((self.mesh.gridEx.size, self.mesh.gridEy.size))
        Ey = np.zeros((self.mesh.gridEx.size, self.mesh.gridEy.size))
        Hz = np.zeros((self.mesh.gridHx.size, self.mesh.gridHy.size))
        
        return Ex, Ey, Hz
    
    def buildFieldsInAllTimeSteps(self, nsteps):
        Ex, Ey, Hz = self.buildFields()
        probeEx = np.zeros((Ex.shape[0], Ex.shape[1], nsteps))
        probeEy = np.zeros((Ey.shape[0], Ey.shape[1], nsteps))
        probeHz = np.zeros((Hz.shape[0], Hz.shape[1], nsteps))
        probeTime = np.zeros(nsteps)
        return probeEx, probeEy, probeHz, probeTime
    
    def initialPulse(self):
        initialH = self.buildFields()[2]
        for i in range(self.mesh.gridHx.size):
            for j in range(self.mesh.gridHy.size):
                initialH[i,j] = math.exp(- ((self.mesh.gridHx[i]-self.center[0])**2 + (self.mesh.gridHy[j]-self.center[1])**2) /math.sqrt(2.0) / self.spread)
        return initialH

    def run(self, nsteps):
        Ex, Ey, Hz = self.buildFields()
        probeEx, probeEy, probeHz, probeTime = self.buildFieldsInAllTimeSteps(nsteps)
        Hz = self.initialPulse()
        t=0.0

        for n in range(nsteps):
            # --- Updates E field ---
            for i in range(1, self.mesh.gridEx.size-1):
                for j in range(1, self.mesh.gridEy.size-1):
                    Ex[i][j] = Ex[i][j] + self.dt/self.dy * (Hz[i][j] - Hz[i  ][j-1])
                    Ey[i][j] = Ey[i][j] - self.dt/self.dx * (Hz[i][j] - Hz[i-1][j  ])
            
            # E field boundary conditions
            
            # PEC
            Ex[ :][ 0] = 0.0
            Ex[ :][-1] = 0.0
            Ey[ 0][ :] = 0.0
            Ey[-1][ :] = 0.0  

            # --- Updates H field. Dey-Mittra ---
            for i in range(self.mesh.gridHx.size):
                for j in range(self.mesh.gridHx.size):
                    if self.mesh.getCellArea((i,j)) != 0:
                        Hz[i][j] = Hz[i][j] - self.dt/self.mesh.getCellArea((i,j)) * (self.mesh.getCellLengths((i,j))["right"]*Ey[i+1][j  ] - self.mesh.getCellLengths((i,j))["left"]*Ey[i][j] +\
                                                                                    self.mesh.getCellLengths((i,j))["bottom"]*Ex[i  ][j] - self.mesh.getCellLengths((i,j))["top"]*Ex[i][j+1])
                    
            
            
            # --- Updates output requests ---
            probeEx[:,:,n] = Ex[:,:]
            probeEy[:,:,n] = Ey[:,:]
            probeHz[:,:,n] = Hz[:,:]
            probeTime[n] = t
            t += self.dt

        return probeEx, probeEy, probeHz, probeTime
