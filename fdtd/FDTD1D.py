import numpy as np
from math import exp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

class FDTD1D_Class():
    def __init__(self, ke, cfl, t0, spread, nsteps):
        self.ke = ke
        self.cfl = cfl
        self.t0 = t0
        self.spread = spread
        self.nsteps = nsteps

        self.dx = 1
        self.dt = self.dx * self.cfl
        self.cb = self.dt/self.dx

    def SpatialMesh(self):
        N = 1 + (self.ke)/self.dx
        return np.linspace(0, self.ke, int(N))

    def buildFields(self):
        ex = np.zeros(self.ke)
        hy = np.zeros(self.ke)

        return ex, hy
    
    def buildFieldsInAllTimeSteps(self):
        probeE = np.zeros((self.ke, self.nsteps+1))
        probeH = np.zeros((self.ke, self.nsteps+1))

        return probeE, probeH
    
    def ElectricGaussianPulse(self, time_step):
        return self.dt*exp(-0.5 * ((self.t0 - self.dt*time_step) / self.spread) ** 2)
    
    def MagneticGaussianPulse(self, time_step):
        return self.dt*exp(-0.5 * ((self.t0 - self.dt/2 - self.dx/2 - self.dt*time_step) / self.spread) ** 2)
    
    def FDTDLoop(self):
        StaticFields = self.buildFields()
        ex = StaticFields[0]
        hy = StaticFields[1]

        TimeFields = self.buildFieldsInAllTimeSteps()
        probeE = TimeFields[0]
        probeH = TimeFields[1]

        kc = int(self.ke / 5)

        for time_step in range(1, self.nsteps + 1):

            for k in range(1, self.ke):
                ex[k] = ex[k] + self.cb * (hy[k - 1] - hy[k])

            ex[kc] += self.ElectricGaussianPulse(time_step)
            hy[kc] += self.MagneticGaussianPulse(time_step)

            for k in range(self.ke - 1):
                hy[k] = hy[k] + self.cb * (ex[k] - ex[k + 1])

            probeE[:,time_step]=ex[:]
            probeH[:,time_step]=hy[:]
            
        return probeE, probeH



## Here below is the comprobation of obtaining the same result as before ## 

# fdtd = FDTD1D_Class(ke=200, cfl=1, t0=40, spread=10, nsteps=1000)
# FinalFields = fdtd.FDTDLoop()
# probeE = FinalFields[0]
# probeH = FinalFields[1]


# # Create a figure and axis for the animation
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
# line1, = ax1.plot(probeE[:,0], color='k', linewidth=1)
# line2, = ax2.plot(probeH[:,0], color='k', linewidth=1)

# # Function to initialize the plot
# def init():
#     ax1.set_ylabel('E$_x$', fontsize='14')
#     ax1.set_xlim(0, 200)
#     ax1.set_ylim(-1.2, 1.2)

#     ax2.set_ylabel('H$_y$', fontsize='14')
#     ax2.set_xlabel('FDTD cells')
#     ax2.set_xlim(0, 200)
#     ax2.set_ylim(-1.2, 1.2)
#     plt.subplots_adjust(bottom=0.2, hspace=0.45)
#     return line1, line2


# def animate(i):
#     line1.set_ydata(probeE[:,i])
#     line2.set_ydata(probeH[:,i])
#     return line1, line2


# # Create the animation
# ani = FuncAnimation(fig, animate, frames=1001, init_func=init, blit=True, interval=10, repeat=False)

# # Show the animation
# plt.show()