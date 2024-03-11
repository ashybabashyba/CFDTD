"""FDTD 1D
Simulation in free space
"""
import numpy as np
from math import exp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

ke = 200
cfl = 1
ex = np.zeros(ke)
hy = np.zeros(ke)

# Pulse parameters
kc = int(ke / 5)
t0 = 40
spread = 10
nsteps = 1000
probeE = np.zeros((ke, nsteps+1))
probeH = np.zeros((ke, nsteps+1))

# Spatial and time steps
epsilon0=1
mu0=1
c0=1/np.sqrt(epsilon0*mu0)
deltax=1
deltat=deltax/c0*cfl
cb=deltat*c0/deltax


# Main FDTD Loop
for time_step in range(1, nsteps + 1):

    # Calculate the Ex field
    for k in range(1, ke):
        ex[k] = ex[k] + cb * (hy[k - 1] - hy[k])

    # Put a Gaussian pulse in the middle
    ex[kc] += deltat*exp(-0.5 * ((t0 - deltat*time_step) / spread) ** 2)
    hy[kc] += deltat*exp(-0.5 * ((t0 - deltat/2 - deltax/2/c0 - deltat*time_step) / spread) ** 2)


    # Calculate the Hy field
    for k in range(ke - 1):
        hy[k] = hy[k] + cb * (ex[k] - ex[k + 1])

    probeE[:,time_step]=ex[:]
    probeH[:,time_step]=hy[:]

# Create a figure and axis for the animation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
line1, = ax1.plot(probeE[:,0], color='k', linewidth=1)
line2, = ax2.plot(probeH[:,0], color='k', linewidth=1)

# Function to initialize the plot
def init():
    ax1.set_ylabel('E$_x$', fontsize='14')
    ax1.set_xlim(0, 200)
    ax1.set_ylim(-1.2, 1.2)

    ax2.set_ylabel('H$_y$', fontsize='14')
    ax2.set_xlabel('FDTD cells')
    ax2.set_xlim(0, 200)
    ax2.set_ylim(-1.2, 1.2)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    return line1, line2


def animate(i):
    line1.set_ydata(probeE[:,i])
    line2.set_ydata(probeH[:,i])
    return line1, line2


# Create the animation
ani = FuncAnimation(fig, animate, frames=nsteps+1, init_func=init, blit=True, interval=10, repeat=False)

# Show the animation
plt.show()