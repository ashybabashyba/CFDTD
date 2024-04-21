import numpy as np
import math
from matplotlib import pyplot as plt
import matplotlib.animation as animation


class Mesh():
    def __init__(self, box_size, dx, dy):
        self.dx = dx
        self.dy = dy
        self.boxSize = box_size

        self.gridEx = np.linspace(0, box_size, int(1 + box_size/dx))
        self.gridEy = np.linspace(0, box_size, int(1 + box_size/dy))

        self.gridHx = (self.gridEx[1:] + self.gridEx[:-1]) / 2.0
        self.gridHy = (self.gridEy[1:] + self.gridEy[:-1]) / 2.0

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

            # --- Updates H field ---
            for i in range(self.mesh.gridHx.size):
                for j in range(self.mesh.gridHx.size):
                    Hz[i][j] = Hz[i][j] - self.dt/self.dx * (Ey[i+1][j  ] - Ey[i][j]) +\
                                          self.dt/self.dy * (Ex[i  ][j+1] - Ex[i][j])
                    
            
            
            # --- Updates output requests ---
            probeEx[:,:,n] = Ex[:,:]
            probeEy[:,:,n] = Ey[:,:]
            probeHz[:,:,n] = Hz[:,:]
            probeTime[n] = t
            t += self.dt

        return probeEx, probeEy, probeHz, probeTime


mesh = Mesh(box_size=10.0, dx=0.1, dy=0.1)
solver = CFDTD2D(mesh, (5.0, 5.0), spread=1.0, cfl=0.99)
nsteps = int(50 / solver.dt)
probeEx, probeEy, probeHz, probeTime = solver.run(nsteps)

# --- Creates animation ---
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#ax = plt.axes(xlim=(gridE[0], gridE[-1]), ylim=(-1.1, 1.1))
ax.set_xlabel('X coordinate [m]')
ax.set_ylabel('Y coordinate [m]')
line = plt.imshow(probeHz[:,:,0], animated=True, vmin=-0.5, vmax=0.5)
timeText = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    line.set_array(probeHz[:,:,0])
    timeText.set_text('')
    return line, timeText

def animate(i):
    line.set_array(probeHz[:,:,i])
    timeText.set_text('Time = %2.1f [s]' % (probeTime[i]))
    return line, timeText

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=nsteps, interval=50, blit=True)

plt.show()