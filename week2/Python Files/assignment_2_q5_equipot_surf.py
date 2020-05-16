from electric_multipoles import ChargeDistribution, e, Dipole
from math import asin, pi 
from mayavi.mlab import contour3d
from ai.cs import sp2cart
import numpy as np

# Radius of Carbon atom
r_C = 70e-12 # m

# Radius of Hydrogen atom
r_H = 53e-12 # m

# Carbon - Hydrogen Bond Length
BL_CH = 160e-12 # m

r_min, r_max = -BL_CH*2, BL_CH*2

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

x_array = np.linspace(r_min, r_max, n_points)
y_array = np.linspace(r_min, r_max, n_points)
z_array = np.linspace(r_min, r_max, n_points)

V_Tensor = np.zeros((n_points, n_points, n_points))

print('Started calculating V_Tensor')

for i in range(n_points) :

    x = x_array[i]

    for j in range(n_points) :

        y = y_array[j]

        for k in range(n_points) :

            z = z_array[k]

            pos = np.array([x, y, z])

            V_Tensor[i][j][k] = C_ChargeDist.V(pos) + H1.V(pos) + H2.V(pos) + H3.V(pos) + H4.V(pos)

print('Finished calculating V_Tensor')

X, Y, Z = np.meshgrid( x_array, y_array, z_array, indexing='ij')

V_Tensor[ np.isinf(V_Tensor) ] = np.nan

contour3d(  X, Y, Z, V_Tensor,
            name='Equipotential surfaces for Methane Molecule', 
            contours=50, opacity=0.5, colormap='magma')

waitkey = input()