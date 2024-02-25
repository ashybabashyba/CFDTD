"""CFDTD 1D
Simulation in free space
"""
import numpy as np
from math import exp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

ke = 200
ex = np.zeros(ke)
hy = np.zeros(ke)

# PEC sheet position (in a decimal)
kp=100.5

# Pulse parameters
kc = int(ke / 4)
t0 = 40
spread = 12
nsteps = 1000

# Spatial and time steps
epsilon0=1
mu0=1
c0=1/np.sqrt(epsilon0*mu0)
deltax=ke
deltat=deltax/(2*c0)
cb=deltat*c0/deltax


# Create a figure and axis for the animation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
line1, = ax1.plot(ex, color='k', linewidth=1)
line2, = ax2.plot(hy, color='k', linewidth=1)
ax1.axvline(x=kp, color='r', linestyle='--', label='kp')
ax2.axvline(x=kp, color='r', linestyle='--', label='kp')

# Function to initialize the plot
def init():
    ax1.set_ylabel('E$_x$', fontsize='14')
    ax1.set_xlim(0, ke)
    ax1.set_ylim(-2.2, 2.2)

    ax2.set_ylabel('H$_y$', fontsize='14')
    ax2.set_xlabel('FDTD cells')
    ax2.set_xlim(0, ke)
    ax2.set_ylim(-2.2, 2.2)
    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    return line1, line2

# Function to update the plot in each animation frame
def update(frame):
    global ex, hy

    # Calculate the Ex field
    for k in range(1, ke):
        ex[k] = ex[k] + cb * (hy[k - 1] - hy[k])

    # Put a Gaussian pulse in the middle
    pulse = exp(-0.5 * ((t0 - frame) / spread) ** 2)
    ex[kc] = pulse


    # Calculate the Hy field
    for k in range(ke - 1):
        if k==np.floor(kp):
            hy[k] = hy[k] + cb * (ex[k])
        elif k==(np.floor(kp)+1):
            hy[k] = hy[k] - cb * (ex[k + 1])
        else:
            hy[k] = hy[k] + cb * (ex[k] - ex[k + 1])

    # Update the plot data
    line1.set_ydata(ex)
    line2.set_ydata(hy)
    
    # Display the time step in the plot
    ax1.text(100, 0.5, 'T = {}'.format(frame), horizontalalignment='center')


    return line1, line2

# Create the animation
ani = FuncAnimation(fig, update, frames=nsteps, init_func=init, blit=True, interval=10, repeat=False)

# Show the animation
plt.show()