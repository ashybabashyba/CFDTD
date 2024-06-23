import numpy as np
import math
from numba import jit
from matplotlib import pyplot as plt
import matplotlib.animation as animation

@jit(nopython=True, parallel=True)
def electricFieldStep(Ex_prev, Ey_prev, Hz_prev, dx, dy, dt, left, bottom, area):
    Ex_next = np.zeros(Ex_prev.shape)
    Ey_next = np.zeros(Ey_prev.shape)
    for i in range(1, Ex_prev.shape[0]-1):
        for j in range(1, Ex_prev.shape[1]-1):
            if not np.isclose(bottom[i,j], 0):
                Ex_next[i][j] = Ex_prev[i][j] + dt/dy * (Hz_prev[i][j] - Hz_prev[i][j-1])
            if not np.isclose(left[i,j], 0):
                Ey_next[i][j] = Ey_prev[i][j] - dt/dx * (Hz_prev[i][j] - Hz_prev[i-1][j])
    
    return Ex_next, Ey_next

@jit(nopython=True, parallel=True)
def magneticFieldStep(Ex_prev, Ey_prev, Hz_prev, dt, area, left, right, top, bottom):
    Hz_next = np.zeros(Hz_prev.shape)
    for i in range(Hz_prev.shape[0]):
        for j in range(Hz_prev.shape[1]):
            if area[i,j] != 0:
                Hz_next[i][j] = Hz_prev[i][j] - dt/area[i,j] * (right[i,j]*Ey_prev[i+1][j] - left[i,j]*Ey_prev[i][j] +\
                                                                bottom[i,j]*Ex_prev[i  ][j] - top[i,j]*Ex_prev[i][j+1])

    return Hz_next    

@jit(nopython=True, parallel=True)
def electricFieldStepNonConformal(Ex_prev, Ey_prev, Hz_prev, dx, dy, dt, left, bottom, area, xmin_index, xmax_index, ymin_index, ymax_index):
    Ex_next = np.zeros(Ex_prev.shape)
    Ey_next = np.zeros(Ey_prev.shape)
    for i in range(1, Ex_prev.shape[0]-1):
        for j in range(1, Ex_prev.shape[1]-1):
            if np.isclose(area[i,j], 1):
                if (j != ymin_index and j !=ymax_index+1) and not np.isclose(bottom[i,j], 0): 
                    Ex_next[i][j] = Ex_prev[i][j] + dt/dy * (Hz_prev[i][j] - Hz_prev[i][j-1])
                if (i != xmin_index and i != xmax_index+1) and not np.isclose(left[i,j], 0):
                    Ey_next[i][j] = Ey_prev[i][j] - dt/dx * (Hz_prev[i][j] - Hz_prev[i-1][j])
    
    return Ex_next, Ey_next

@jit(nopython=True, parallel=True)
def magneticFieldStepNonConformal(Ex_prev, Ey_prev, Hz_prev, dt, area, left, right, top, bottom):
    Hz_next = np.zeros(Hz_prev.shape)
    for i in range(Hz_prev.shape[0]):
        for j in range(Hz_prev.shape[1]):
            if np.isclose(area[i,j], 1):
                Hz_next[i][j] = Hz_prev[i][j] - dt/area[i,j] * (right[i,j]*Ey_prev[i+1][j] - left[i,j]*Ey_prev[i][j] +\
                                                                bottom[i,j]*Ex_prev[i  ][j] - top[i,j]*Ex_prev[i][j+1])

    return Hz_next 

class CFDTD2D():
    def __init__(self, mesh, initialPulse, cfl, solver_type=None):
        self.mesh = mesh
        self.pulse = initialPulse
        self.center = self.pulse.center
        self.cfl = cfl
        self.spread = self.pulse.spread

        self.SolverType = solver_type

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
        Ex, Ey, Hz = self.pulse.buildPulse()
        t=0.0

        cell_area = np.array([[self.mesh.getCellArea((i, j)) for j in range(self.mesh.gridHy.size)] for i in range(self.mesh.gridHx.size)])
        cell_lengths_right = np.array([[self.mesh.getCellLengths((i, j))["right"] for j in range(self.mesh.gridHy.size)] for i in range(self.mesh.gridHx.size)])
        cell_lengths_left = np.array([[self.mesh.getCellLengths((i, j))["left"] for j in range(self.mesh.gridHy.size)] for i in range(self.mesh.gridHx.size)])
        cell_lengths_bottom = np.array([[self.mesh.getCellLengths((i, j))["bottom"] for j in range(self.mesh.gridHy.size)] for i in range(self.mesh.gridHx.size)])
        cell_lengths_top = np.array([[self.mesh.getCellLengths((i, j))["top"] for j in range(self.mesh.gridHy.size)] for i in range(self.mesh.gridHx.size)])

        xmin, xmax, ymin, ymax = self.mesh.getMinMaxIndexInsideNonConformalCells()

        for n in range(nsteps):

            if self.SolverType == "Non Conformal":
                Hz = magneticFieldStepNonConformal(Ex_prev=Ex, Ey_prev=Ey, Hz_prev=Hz, dt=self.dt, area=cell_area, left=cell_lengths_left, right=cell_lengths_right, top=cell_lengths_top, bottom=cell_lengths_bottom)
            else:
                Hz = magneticFieldStep(Ex_prev=Ex, Ey_prev=Ey, Hz_prev=Hz, dt=self.dt, area=cell_area, left=cell_lengths_left, right=cell_lengths_right, top=cell_lengths_top, bottom=cell_lengths_bottom)
            
            if self.SolverType == "Non Conformal":
                Ex, Ey = electricFieldStepNonConformal(Ex_prev=Ex, Ey_prev=Ey, Hz_prev=Hz, dx=self.dx, dy=self.dy, dt=self.dt, area= cell_area,left=cell_lengths_left, bottom=cell_lengths_bottom, xmin_index=xmin, xmax_index=xmax, ymin_index=ymin, ymax_index=ymax)
            else:
                Ex, Ey = electricFieldStep(Ex_prev=Ex, Ey_prev=Ey, Hz_prev=Hz, dx=self.dx, dy=self.dy, dt=self.dt, area= cell_area,left=cell_lengths_left, bottom=cell_lengths_bottom)
                   
            Ex[ :][ 0] = 0.0
            Ex[ :][-1] = 0.0
            Ey[ 0][ :] = 0.0
            Ey[-1][ :] = 0.0 
             
            probeEx[:,:,n] = Ex[:,:]
            probeEy[:,:,n] = Ey[:,:]
            probeHz[:,:,n] = Hz[:,:]
            probeTime[n] = t
            t += self.dt
            # print ("Time step: %d of %d" % (n, nsteps-1))

        return probeEx, probeEy, probeHz, probeTime
    
    def plotMagneticFieldAnimation(self, nsteps):
        probeEx, probeEy, probeHz, probeTime = self.run(nsteps)
        paused = [False]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim([self.mesh.gridHx[0], self.mesh.gridHx[-1]/self.dx])
        ax.set_ylim([self.mesh.gridHy[0], self.mesh.gridHy[-1]/self.dy])
        ax.set_title('Magnetic field $H_z$')
        ax.set_xlabel('X coordinate [m]')
        ax.set_ylabel('Y coordinate [m]')
        line = plt.imshow(np.transpose(probeHz[:,:,0]), animated=True, vmin=-0.5, vmax=0.5)
        timeText = ax.text(0.02, 0.95, '')

        def init():
            line.set_array(np.transpose(probeHz[:,:,0]))
            timeText.set_text('')
            return line, timeText

        def animate(i):
            if not paused[0]:
                line.set_array(np.transpose(probeHz[:,:,i]))
                timeText.set_text('Time = %2.1f [ns]' % (probeTime[i]))
            return line, timeText

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=nsteps, interval=50, blit=True)
        
        def onClick(event):
            if event.key == 'p':
                paused[0] = not paused[0]

        fig.canvas.mpl_connect('key_press_event', onClick)

        plt.colorbar()
        plt.show()

    def plotElectricFieldXAnimation(self, nsteps):
        probeEx, probeEy, probeHz, probeTime = self.run(nsteps)
        paused = [False]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim([self.mesh.gridEx[0], self.mesh.gridEx[-1]/self.dx])
        ax.set_ylim([self.mesh.gridEy[0], self.mesh.gridEy[-1]/self.dy])
        ax.set_title('Electric field $E_x$')
        ax.set_xlabel('X coordinate [m]')
        ax.set_ylabel('Y coordinate [m]')
        line = plt.imshow(np.transpose(probeEx[:,:,0]), animated=True, vmin=-0.5, vmax=0.5)
        timeText = ax.text(0.02, 0.95, '')

        def init():
            if not paused[0]:
                line.set_array(np.transpose(probeEx[:,:,0]))
                timeText.set_text('')
            return line, timeText

        def animate(i):
            line.set_array(np.transpose(probeEx[:,:,i]))
            timeText.set_text('Time = %2.1f [ns]' % (probeTime[i]))
            return line, timeText

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=nsteps, interval=50, blit=True)
        
        def onClick(event):
            if event.key == 'p':
                paused[0] = not paused[0]

        fig.canvas.mpl_connect('key_press_event', onClick)

        plt.colorbar()
        plt.show()

    def plotElectricFieldYAnimation(self, nsteps):
        probeEx, probeEy, probeHz, probeTime = self.run(nsteps)
        paused = [False]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim([self.mesh.gridEx[0], self.mesh.gridEx[-1]/self.dx])
        ax.set_ylim([self.mesh.gridEy[0], self.mesh.gridEy[-1]/self.dy])
        ax.set_title('Electric field $E_y$')
        ax.set_xlabel('X coordinate [m]')
        ax.set_ylabel('Y coordinate [m]')
        line = plt.imshow(np.transpose(probeEy[:,:,0]), animated=True, vmin=-0.5, vmax=0.5)
        timeText = ax.text(0.02, 0.95, '')

        def init():
            if not paused[0]:
                line.set_array(np.transpose(probeEy[:,:,0]))
                timeText.set_text('')
            return line, timeText

        def animate(i):
            line.set_array(np.transpose(probeEy[:,:,i]))
            timeText.set_text('Time = %2.1f [ns]' % (probeTime[i]))
            return line, timeText

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=nsteps, interval=50, blit=True)
        
        def onClick(event):
            if event.key == 'p':
                paused[0] = not paused[0]

        fig.canvas.mpl_connect('key_press_event', onClick)

        plt.colorbar()
        plt.show()

    def plotMagneticFieldFrame(self, frame):
        nsteps = int(frame/self.dt)
        probeEx, probeEy, probeHz, probeTime = self.run(nsteps+1)

        plt.imshow(np.transpose(probeHz[:, :, nsteps]), cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f'Magnetic field $H_z$ at {probeTime[nsteps]:.2f} ns')
        plt.xlabel('X coordinate [m]')
        plt.ylabel('Y coordinate [m]')
        plt.show()

    def plotElectricFieldXFrame(self, frame):
        nsteps = int(frame/self.dt)
        probeEx, probeEy, probeHz, probeTime = self.run(nsteps+1)

        plt.imshow(np.transpose(probeEx[:, :, nsteps]), cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f'Electric field $E_x$ at {probeTime[nsteps]:.2f} ns')
        plt.xlabel('X coordinate [m]')
        plt.ylabel('Y coordinate [m]')
        plt.show()

    def plotElectricFieldYFrame(self, frame):
        nsteps = int(frame/self.dt)
        probeEx, probeEy, probeHz, probeTime = self.run(nsteps+1)

        plt.imshow(np.transpose(probeEy[:, :, nsteps]), cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f'Electric field $E_y$ at {probeTime[nsteps]:.2f} ns')
        plt.xlabel('X coordinate [m]')
        plt.ylabel('Y coordinate [m]')
        plt.show()