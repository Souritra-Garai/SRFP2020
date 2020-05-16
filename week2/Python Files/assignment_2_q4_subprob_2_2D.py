from electric_multipoles import ChargeDistribution, e
from ai.cs import sp2cart
import numpy as np
from ai.cs import sp2cart
from math import asin, pi 

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

a = 38e-12 # m

n = 3 # order of multipole

r_min, r_max = 10*a, 15*a

n_frames = 50
n_points = 20
n_plots = 5

chargeArray = [4*e] + [-e]*4

tet_theta = - asin(1/3)

chargePosArray = np.array( sp2cart([0] + [a]*4, [0, pi/2] + [tet_theta] * 3, [0, 0, -pi/3, pi/3, pi]) ).transpose()

myChargeDist = ChargeDistribution(chargeArray, chargePosArray)

r_array     = np.linspace( r_min, r_max, n_points)
theta_array = np.linspace( tet_theta, pi / 2, n_points)
phi_array   = np.linspace(-pi / 3, pi / 3, n_frames)

V_Tensor = np.zeros((n_frames, n_plots, n_points))
R_Tensor = np.zeros_like(V_Tensor)

print('Started calculating V_Tensor')

for i in range(n_frames) :

    phi = phi_array[i]

    for j in range(n_plots) :

        theta = theta_array[j]

        for k in range(n_points) :

            r = r_array[k]

            pos = np.array(sp2cart(r, theta, phi))

            V_Tensor[i][j][k] = myChargeDist.V(pos)

            R_Tensor[i][j][k] = V_Tensor[i][j][0] * ( r_array[0] ** (n + 1) ) / ( r ** (n + 1) )

print('Finished calculating V_Tensor')

fig = plt.figure()
ax1 = plt.axes(xlim=(r_min, r_max), ylim=(np.min(V_Tensor), np.max(V_Tensor)))

line, = ax1.plot([], [], lw=2)

plt.xlabel(r'$r$ in m')
plt.ylabel(r'$V$ in volts')

lines = []

for i in range(n_plots) :

    lobj = ax1.plot([], [], lw=10, label=('Actual V ' + str(i)))[0]
    lines.append(lobj)

    lobj = ax1.plot([], [], lw=2, label=('Variation with R ' + str(i)))[0]
    lines.append(lobj)

plt.legend()

def init() :

    for line in lines:

        line.set_data([],[])

    return lines

def update(i) :

    for j in range(n_plots) :

        lines[2*j].set_data(r_array, V_Tensor[i][j])
        lines[2*j + 1].set_data(r_array, R_Tensor[i][j])

    return lines

myAnimation = FuncAnimation(fig, update, init_func=init, frames=n_frames, blit=True)
plt.show()