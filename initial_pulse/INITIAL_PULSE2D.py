import numpy as np
import math
from matplotlib import pyplot as plt

class InitialPulse():
    def __init__(self, mesh, initial_position, spread, pulse_type, m=None, n=None):
        self.center = initial_position
        self.spread = spread
        self.type = pulse_type
        self.mesh = mesh

        self.m = m
        self.n = n

        self.Ex = np.zeros((self.mesh.gridEx.size, self.mesh.gridEy.size))
        self.Ey = np.zeros((self.mesh.gridEx.size, self.mesh.gridEy.size))
        self.Hz = np.zeros((self.mesh.gridHx.size, self.mesh.gridHy.size))

    def buildPulse(self):
        if self.type == "Magnetic Gaussian":
            self.Hz = magneticGaussian(H0=self.Hz, mesh=self.mesh, center=self.center, spread=self.spread)
        elif self.type == "Rectangular Resonant Cavity":
            self.Hz = rectangularResonantCavity(H0=self.Hz, mesh=self.mesh, m=self.m, n=self.n)
        elif self.type == "Non Conformal Rectangular Resonant Cavity":
            self.Hz = rectangularResonantCavityNonConformal(H0=self.Hz, mesh=self.mesh, m=self.m, n=self.n)
        elif self.type == "MagneticY Gaussian":
            self.Hz = electricYGaussian(H0=self.Hz, mesh=self.mesh, center=self.center, spread=self.spread)
        else:
            raise ValueError('Pulse type not defined')
        return self.Ex, self.Ey, self.Hz
    
def magneticGaussian(H0, mesh, center, spread):
    initialH = np.zeros(H0.shape)
    for i in range(mesh.gridHx.size):
        for j in range(mesh.gridHy.size):
            initialH[i,j] = math.exp(- ((mesh.gridHx[i]-center[0])**2 + (mesh.gridHy[j]-center[1])**2) /math.sqrt(2.0) / spread)     
    return initialH

def rectangularResonantCavity(H0, mesh, m, n):
    initialH = np.zeros(H0.shape)
    conformalCells, outsideNonConformalCells, insideNonConformalCells = mesh.getCellSeparationByType()
    a = np.abs(mesh.nodesList[0][0] - mesh.nodesList[1][0])
    b = np.abs(mesh.nodesList[0][1] - mesh.nodesList[3][1])
    x0 = mesh.nodesList[0][0]
    y0 = mesh.nodesList[0][1]
    for i in range(mesh.gridHx.size):
        for j in range(mesh.gridHy.size):
            if (i,j) not in outsideNonConformalCells:
                initialH[i,j] = np.cos(m*np.pi*(mesh.gridHx[i] - x0)/a)*np.cos(n*np.pi*(mesh.gridHy[j] - y0)/b)
    return initialH

def rectangularResonantCavityNonConformal(H0, mesh, m, n):
    initialH = np.zeros(H0.shape)
    conformalCells, outsideNonConformalCells, insideNonConformalCells = mesh.getCellSeparationByType()
    a = np.abs(mesh.nodesList[0][0] - mesh.nodesList[1][0])
    b = np.abs(mesh.nodesList[0][1] - mesh.nodesList[3][1])

    xmin, xmax, ymin, ymax = mesh.getMinMaxIndexInsideNonConformalCells()
    dx_l = mesh.getCellLengths((xmin-1, ymin))["bottom"]
    dx_r = mesh.getCellLengths((xmax+1, ymin))["bottom"]
    dy_b = mesh.getCellLengths((xmin, ymin-1))["left"]
    dy_t = mesh.getCellLengths((xmin, ymax+1))["left"]

    x0 = mesh.nodesList[0][0]
    y0 = mesh.nodesList[0][1]
    for i in range(mesh.gridHx.size):
        for j in range(mesh.gridHy.size):
            if (i,j) in insideNonConformalCells:
                initialH[i,j] = np.cos(m*np.pi*(mesh.gridHx[i] - x0 - dx_l)/(a-dx_l-dx_r))*np.cos(n*np.pi*(mesh.gridHy[j] - y0 - dy_b)/(b-dy_b-dy_t))
    return initialH

def electricYGaussian(H0, mesh, center, spread):
    initialH = np.zeros(H0.shape)
    centerX = center[0]
    centerY = center[1]
    conformalCells, outsideNonConformalCells, insideNonConformalCells = mesh.getCellSeparationByType()
    for i in range(mesh.gridHx.size):
        for j in range(mesh.gridHy.size):
            if (i, j) not in outsideNonConformalCells:
                initialH[i, j] = math.exp(- ((mesh.gridHx[i]-centerX)**2 ) /math.sqrt(2.0) / spread)
    return initialH