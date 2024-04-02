import numpy as np
from math import exp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

class Mesh():
    def __init__(self, box_size, dx):
        kpi = np.floor(box_size)
                
        V1 = np.linspace(0, kpi, int(1 + kpi/dx))
        V2 = np.array([box_size])
        if kpi == box_size:
            self.vx = V1
        else:
            self.vx = np.concatenate((V1, V2))

    def getDX(self):
        return self.vx[1] - self.vx[0]
    
    def getLength(self):
        return self.vx[-1]
    
    def getLeftover(self):
        kp = self.getLength()
        if self.kp==np.floor(kp):
            return 1.0
        else:
            return 1.0 /(kp-np.floor(kp))
    
    def numberOfCells(self):
        return len(self.vx) - 1


class CFDTD1D():
    def __init__(self, mesh, cfl, t0, spread):
        
        self.mesh = mesh
        
        self.cfl = cfl
        self.t0 = t0
        self.spread = spread

        self.dt = self.mesh.getDX() * self.cfl
        self.cb = self.dt/self.mesh.getDX()
    
    def buildFields(self):
        ex = np.zeros(self.mesh.numberOfCells())
        hy = np.zeros(self.mesh.numberOfCells())

        return ex, hy
    
    def buildFieldsInAllTimeSteps(self, nsteps):
        ex, hy = self.buildFields()
        probeE = np.zeros((len(ex), nsteps))
        probeH = np.zeros((len(hy), nsteps))

        return probeE, probeH
    
    def ElectricGaussianPulse(self, time_step):
        return self.dt*exp(-0.5 * ((self.t0 - self.dt*time_step) / self.spread) ** 2)
    
    def MagneticGaussianPulse(self, time_step):
        return self.dt*exp(-0.5 * ((self.t0 - self.dt/2 - self.dx/2 - self.dt*time_step) / self.spread) ** 2)
    
    def run(self, nsteps):
        ex, hy = self.buildFields()
        probeE, probeH = self.buildFieldsInAllTimeSteps(nsteps)
        
        kc = int(self.ke / 5)

        leftover = self.mesh.getLeftover()

        for time_step in range(nsteps):

            for k in range(1, self.numberOfCells()):
                if k<np.floor(self.kp):
                    ex[k] = ex[k] + self.cb * (hy[k - 1] - hy[k])
                elif k==np.floor(self.kp):
                    ex[k] = ex[k] + leftover * self.cb * (hy[k - 1] - hy[k])

            ex[kc] += self.ElectricGaussianPulse(time_step)
            hy[kc] += self.MagneticGaussianPulse(time_step)

            for k in range(self.ke - 1):
                if k==np.floor(self.kp):
                    hy[k] = hy[k] + leftover * self.cb * (ex[k])
                elif k<np.floor(self.kp):
                    hy[k] = hy[k] + self.cb * (ex[k] - ex[k + 1])

            probeE[:,time_step]=ex[:]
            probeH[:,time_step]=hy[:]
            
        return probeE, probeH



## Here below is the comprobation of obtaining the same result as before ## 

# fdtd = CFDTD1D_Class(ke=200, kp=100.5, cfl=0.75, t0=40, spread=10, nsteps=1000)
# FinalFields = fdtd.CFDTDLoop()
# probeE = FinalFields[0]
# probeH = FinalFields[1]


# # Create a figure and axis for the animation
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
# line1, = ax1.plot(probeE[:,0], color='k', linewidth=1)
# line2, = ax2.plot(probeH[:,0], color='k', linewidth=1)
# ax1.axvline(x=100.5, color='r', linestyle='--', label='kp')
# ax2.axvline(x=100.5, color='r', linestyle='--', label='kp')

# # Function to initialize the plot
# def init():
#     ax1.set_ylabel('E$_x$', fontsize='14')
#     ax1.set_xlim(0, 200)
#     ax1.set_ylim(-2.2, 2.2)

#     ax2.set_ylabel('H$_y$', fontsize='14')
#     ax2.set_xlabel('FDTD cells')
#     ax2.set_xlim(0, 200)
#     ax2.set_ylim(-2.2, 2.2)
#     plt.subplots_adjust(bottom=0.2, hspace=0.45)
#     return line1, line2


# def animate(i):
#     line1.set_ydata(probeE[:,i])
#     line2.set_ydata(probeH[:,i])
#     return line1, line2


# # Create the animation
# ani = FuncAnimation(fig, animate, frames=1001, init_func=init, blit=True, interval=2, repeat=False)

# # Show the animation
# plt.show()