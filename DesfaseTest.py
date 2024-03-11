"""CFDTD 1D
Simulation in free space
"""
import numpy as np
from math import exp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

ke = 120
ex1 = np.zeros(ke)
hy1 = np.zeros(ke)

ex2 = np.zeros(ke)
hy2 = np.zeros(ke)

# PEC sheet position (in a decimal) and CFL
kp=119.50
cfl = 0.4

if kp==np.floor(kp):
    leftover=1
else:
    leftover=1/(kp-np.floor(kp))

if (cfl*leftover)>1:
    print("No se satisface la condición de Courant (?)")

# Pulse parameters
kc = int(ke / 5)
t0 = 40
spread = 10
nsteps = 10000

# Spatial and time steps
epsilon0=1
mu0=1
c0=1/np.sqrt(epsilon0*mu0)
deltax=1
deltat=deltax/c0*cfl
cb=deltat*c0/deltax


# Create a figure and axis for the animation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
line1, = ax1.plot(ex1, color='k', linewidth=1)
line2, = ax2.plot(hy1, color='k', linewidth=1)
line3, = ax1.plot(ex2, color='b', linewidth=1)
line4, = ax2.plot(hy2, color='b', linewidth=1)
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
    return line1, line2, line3, line4

# Function to update the plot in each animation frame
def update(frame):
    global ex1, hy1, ex2, hy2

    # Calculate the Ex field
    for k in range(1, ke):
        if k<np.floor(kp):
            ex1[k] = ex1[k] + cb * (hy1[k - 1] - hy1[k])
        elif k==np.floor(kp):
            ex1[k] = ex1[k] + leftover*cb * (hy1[k - 1] - hy1[k])

        ex2[k] = ex2[k] + cb * (hy2[k - 1] - hy2[k])


    # Put a Gaussian pulse in the middle
    ex1[kc] += deltat*exp(-0.5 * ((t0 - deltat*frame) / spread) ** 2)
    hy1[kc] += deltat*exp(-0.5 * ((t0 - deltat/2 - deltax/2/c0 - deltat*frame) / spread) ** 2)

    ex2[kc] += deltat*exp(-0.5 * ((t0 - deltat*frame) / spread) ** 2)
    hy2[kc] += deltat*exp(-0.5 * ((t0 - deltat/2 - deltax/2/c0 - deltat*frame) / spread) ** 2)


    # Calculate the Hy field
    for k in range(1, ke):
        if k==np.floor(kp):
            hy1[k] = hy1[k] + leftover*cb * (ex1[k])
        elif k<np.floor(kp):
            hy1[k] = hy1[k] + cb * (ex1[k] - ex1[k + 1])

    for k in range(ke - 1):
        hy2[k] = hy2[k] + cb * (ex2[k] - ex2[k + 1])


    # Update the plot data
    line1.set_ydata(ex1)
    line2.set_ydata(hy1)
    line3.set_ydata(ex2)
    line4.set_ydata(hy2)
    
    # Display the time step in the plot
    ax1.text(100, 0.5, 'T = {}'.format(frame), horizontalalignment='center')


    return line1, line2, line3, line4

# Create the animation
ani = FuncAnimation(fig, update, frames=nsteps, init_func=init, blit=True, interval=2, repeat=False)

# Show the animation
plt.show()