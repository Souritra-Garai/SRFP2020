from electric_multipoles import ChargeDistribution, e
from mayavi.mlab import contour3d
from ai.cs import sp2cart
from math import asin, pi
import numpy as np

a = 38e-12 # m

r_min, r_max = -2*a, 2*a

chargeArray = [4*e] + [-e]*4

tet_theta = - asin(1/3)

chargePosArray = np.array( sp2cart([0] + [a]*4, [0, pi/2] + [tet_theta] * 3, [0, 0, -pi/3, pi/3, pi]) ).transpose()

myChargeDist = ChargeDistribution(chargeArray, chargePosArray)

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

            V_Tensor[i][j][k] = myChargeDist.V( np.array([x,y,z]) )

print('Finished calculating V_Tensor')

X, Y, Z = np.meshgrid( x_array, y_array, z_array, indexing='ij')

V_Tensor[ np.isinf(V_Tensor) ] = np.nan

contour3d(  X, Y, Z, V_Tensor,
            name='Equipotential surfaces for Tetrahedral Charge Distribution', 
            contours=50, opacity=0.5, colormap='magma')

waitkey = input()