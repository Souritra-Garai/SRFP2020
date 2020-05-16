from electric_multipoles import ChargeDistribution, e
import numpy as np
from ai.cs import sp2cart
from math import asin, pi 

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt

a = 38e-12 # m

n_frames = 50
frequency = 10 # Hz
r_min, r_max = a*10, a*15

chargeArray = [4*e] + [-e]*4

tet_theta = - asin(1/3)

chargePosArray = np.array( sp2cart([0] + [a]*4, [0, pi/2] + [tet_theta] * 3, [0, 0, -pi/3, pi/3, pi]) ).transpose()

myChargeDist = ChargeDistribution(chargeArray, chargePosArray)

n_points = 20

r_array     = np.linspace( r_min, r_max, n_points)
theta_array = np.linspace( tet_theta, pi / 2, n_points)
phi_array   = np.linspace(-pi / 3, pi / 3, n_frames)

V_Tensor = np.zeros((n_frames, n_points, n_points))

print('Started calculating V_Tensor')

for i in range(n_frames) :

    phi = phi_array[i]

    for j in range(n_points) :

        r = r_array[j]

        for k in range(n_points) :

            theta = theta_array[k]

            pos = np.array(sp2cart(r, theta, phi))

            V_Tensor[i][j][k] = myChargeDist.V(pos)

print('Finished calculating V_Tensor')


X, Y = np.meshgrid(r_array, theta_array * 180 / np.pi, indexing='ij')

fig = plt.figure(figsize=(16,10))
ax = fig.gca(projection='3d')
surf = [ ax.plot_surface(X, Y, V_Tensor[0], cmap='magma', rstride=1, cstride=1) ]

def update(i) :

    Axes3D.set_title(ax, label=(r'$\phi$ = ' + str(round(phi_array[i] * 180 / np.pi)) + r'$^o$'))

    surf[0].remove()
    surf[0] = ax.plot_surface(X, Y, V_Tensor[i], cmap='magma')

    return surf

ax.set_xlabel(r'$r$ in m')
ax.set_ylabel(r'$\theta$ in degrees')
ax.set_zlabel(r'$V$ in volts')

# Customize the z axis.
ax.set_zlim(np.amin(V_Tensor)*1.2, np.max(V_Tensor)*1.2)
ax.zaxis.set_major_formatter(FormatStrFormatter('   %1.0e'))

# Add a color bar which maps values to colors.
fig.colorbar(surf[0], shrink=0.5, aspect=5)

myAnimation = FuncAnimation(fig, update, n_frames)
plt.show()