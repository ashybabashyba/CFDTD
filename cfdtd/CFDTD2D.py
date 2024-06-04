import numpy as np
import math
from numba import jit
from matplotlib import pyplot as plt
import matplotlib.animation as animation

@jit(nopython=True, parallel=True)
def electricFieldStep(Ex_prev, Ey_prev, Hz_prev, dx, dy, dt):
    Ex_next = np.zeros(Ex_prev.shape)
    Ey_next = np.zeros(Ey_prev.shape)
    for i in range(1, Ex_prev.shape[0]-1):
        for j in range(1, Ex_prev.shape[1]-1):
            Ex_next[j][i] = Ex_prev[j][i] + dt/dy * (Hz_prev[j][i] - Hz_prev[j  ][i-1])
            Ey_next[j][i] = Ey_prev[j][i] - dt/dx * (Hz_prev[j][i] - Hz_prev[j-1][i  ])
    
    return Ex_next, Ey_next

@jit(nopython=True, parallel=True)
def magneticFieldStep(Ex_prev, Ey_prev, Hz_prev, dt, area, left, right, top, bottom):
    Hz_next = np.zeros(Hz_prev.shape)
    for i in range(Hz_prev.shape[0]):
        for j in range(Hz_prev.shape[1]):
            if area[i,j] != 0:
                Hz_next[j][i] = Hz_prev[j][i] - dt/area[i,j] * (right[i,j]*Ey_prev[j+1][i  ] - left[i,j]*Ey_prev[j][i] +\
                                                                bottom[i,j]*Ex_prev[j  ][i] - top[i,j]*Ex_prev[j][i+1])

    return Hz_next    

class InitialPulse():
    def __init__(self, initial_position, spread, pulse_type):
        self.center = initial_position
        self.spread = spread
        self.type = pulse_type

    def magneticGaussian(self, H0, mesh):
        initialH = np.zeros(H0.shape)
        for j in range(mesh.gridHx.size):
            for i in range(mesh.gridHy.size):
                initialH[i,j] = math.exp(- ((mesh.gridHx[i]-self.center[0])**2 + (mesh.gridHy[j]-self.center[1])**2) /math.sqrt(2.0) / self.spread)
        
        return initialH
        

    def pulse(self, dx, dt, step):
        if self.type == "Magnetic Gaussian":
            E0 = dt*np.exp(-0.5 * ((self.initialTime - dt*step) / self.spread) ** 2)
            H0 = dt*np.exp(-0.5 * ((self.initialTime - dt/2 - dx/2 - dt*step) / self.spread) ** 2)
            return E0, H0
        else:
            raise ValueError("Pulse not defined")

class CFDTD2D():
    def __init__(self, mesh, initialPulse, cfl):
        self.mesh = mesh
        self.pulse = initialPulse
        self.center = self.pulse.center
        self.cfl = cfl
        self.spread = self.pulse.spread

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

    def run(self, nsteps):
        Ex, Ey, Hz = self.buildFields()
        probeEx, probeEy, probeHz, probeTime = self.buildFieldsInAllTimeSteps(nsteps)

        if self.pulse.type == 'Magnetic Gaussian':
            Hz = self.pulse.magneticGaussian(Hz, self.mesh)
        else:
            raise ValueError('Pulse type not defined')

        t=0.0

        cell_area = np.array([[self.mesh.getCellArea((i, j)) for j in range(self.mesh.gridHy.size)] for i in range(self.mesh.gridHx.size)])
        cell_lengths_right = np.array([[self.mesh.getCellLengths((i, j))["right"] for j in range(self.mesh.gridHy.size)] for i in range(self.mesh.gridHx.size)])
        cell_lengths_left = np.array([[self.mesh.getCellLengths((i, j))["left"] for j in range(self.mesh.gridHy.size)] for i in range(self.mesh.gridHx.size)])
        cell_lengths_bottom = np.array([[self.mesh.getCellLengths((i, j))["bottom"] for j in range(self.mesh.gridHy.size)] for i in range(self.mesh.gridHx.size)])
        cell_lengths_top = np.array([[self.mesh.getCellLengths((i, j))["top"] for j in range(self.mesh.gridHy.size)] for i in range(self.mesh.gridHx.size)])


        for n in range(nsteps):
            Ex, Ey = electricFieldStep(Ex_prev=Ex, Ey_prev=Ey, Hz_prev=Hz, dx=self.dx, dy=self.dy, dt=self.dt)

            Ex[ :][ 0] = 0.0
            Ex[ :][-1] = 0.0
            Ey[ 0][ :] = 0.0
            Ey[-1][ :] = 0.0  

            Hz = magneticFieldStep(Ex_prev=Ex, Ey_prev=Ey, Hz_prev=Hz, dt=self.dt, area=cell_area, left=cell_lengths_left, right=cell_lengths_right, top=cell_lengths_top, bottom=cell_lengths_bottom)       
            
            probeEx[:,:,n] = Ex[:,:]
            probeEy[:,:,n] = Ey[:,:]
            probeHz[:,:,n] = Hz[:,:]
            probeTime[n] = t
            t += self.dt
            # print ("Time step: %d of %d" % (n, nsteps-1))

        return probeEx, probeEy, probeHz, probeTime
    
    def plotAnimation(self, nsteps):
        probeEx, probeEy, probeHz, probeTime = self.run(nsteps)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim([self.mesh.gridEx[0], self.mesh.gridEx[-1]])
        ax.set_ylim([self.mesh.gridEy[0], self.mesh.gridEy[-1]])
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