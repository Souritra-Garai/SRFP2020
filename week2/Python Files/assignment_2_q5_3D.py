from electric_multipoles import ChargeDistribution, e, Dipole
import numpy as np
from ai.cs import sp2cart
from math import asin, pi 

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt

# Radius of Carbon atom
r_C = 70e-12 # m

# Radius of Hydrogen atom
r_H = 53e-12 # m

# Carbon - Hydrogen Bond Length
BL_CH = 160e-12 # m

n_frames = 50
r_min, r_max = BL_CH*10, BL_CH*15

# Tetrahedral angle (109.4 deg) - 90 deg
tet_theta = - asin(1/3)

# Charge Districution for C atom
C_chargeArray = [4*e] + [-e]*4

C_chargePosArray = np.array( sp2cart([0] + [r_C]*4, [0, pi/2] + [tet_theta] * 3, [0, 0, -pi/3, pi/3, pi]) ).transpose()

C_ChargeDist = ChargeDistribution(C_chargeArray, C_chargePosArray)

# Array of positions of H atoms
H_PosArray = np.array( sp2cart( [BL_CH]*4 , [pi/2] + [tet_theta] * 3, [0, -pi/3, pi/3, pi] ) ).transpose()

# Array of positions of electron of H atom
# assuming them to be farther away from the C atom
H_e_PosArray = np.array( sp2cart( [BL_CH + r_H]*4 , [pi/2] + [tet_theta] * 3, [0, -pi/3, pi/3, pi] ) ).transpose()

# H atom Dipoles
H1 = Dipole.fromChargeDistro(ChargeDistribution([e, -e], [ H_PosArray[0], H_e_PosArray[0]]))
H2 = Dipole.fromChargeDistro(ChargeDistribution([e, -e], [ H_PosArray[1], H_e_PosArray[1]]))
H3 = Dipole.fromChargeDistro(ChargeDistribution([e, -e], [ H_PosArray[2], H_e_PosArray[2]]))
H4 = Dipole.fromChargeDistro(ChargeDistribution([e, -e], [ H_PosArray[3], H_e_PosArray[3]]))

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

            V_Tensor[i][j][k] = C_ChargeDist.V(pos) + H1.V(pos) + H2.V(pos) + H3.V(pos) + H4.V(pos)

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