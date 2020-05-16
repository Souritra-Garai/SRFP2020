from electric_multipoles import ChargeDistribution, e
from ai.cs import sp2cart
from math import asin, pi
import numpy as np

a = 38e-12 # m

r_o = np.array([0, 0, 0]) * 1e-10

chargeArray = [4*e] + [-e]*4

tet_theta = - asin(1/3)

chargePosArray = np.array( sp2cart([0] + [a]*4, [0, pi/2] + [tet_theta] * 3, [0, 0, -pi/3, pi/3, pi]) ).transpose()

myChargeDist = ChargeDistribution(chargeArray, chargePosArray)

print('\nP_e : ')
print(myChargeDist.P_e(r_o))

print('\nq_e : ')
print(myChargeDist.q_e(r_o))

print('\nQ_e : ')
print(myChargeDist.Q_e(r_o))