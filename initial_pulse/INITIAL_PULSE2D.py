import numpy as np
import math
from matplotlib import pyplot as plt

class InitialPulse():
    def __init__(self, mesh, initial_position, spread, pulse_type):
        self.center = initial_position
        self.spread = spread
        self.type = pulse_type
        self.mesh = mesh

        self.Ex = np.zeros((self.mesh.gridEx.size, self.mesh.gridEy.size))
        self.Ey = np.zeros((self.mesh.gridEx.size, self.mesh.gridEy.size))
        self.Hz = np.zeros((self.mesh.gridHx.size, self.mesh.gridHy.size))

    def buildPulse(self):
        if self.type == "Magnetic Gaussian":
            self.Hz = magneticGaussian(H0=self.Hz, mesh=self.mesh, center=self.center, spread=self.spread)
        else:
            raise ValueError('Pulse type not defined')
        return self.Ex, self.Ey, self.Hz
    
def magneticGaussian(H0, mesh, center, spread):
    initialH = np.zeros(H0.shape)
    for j in range(mesh.gridHx.size):
        for i in range(mesh.gridHy.size):
            initialH[i,j] = math.exp(- ((mesh.gridHx[i]-center[0])**2 + (mesh.gridHy[j]-center[1])**2) /math.sqrt(2.0) / spread)     
    return initialH