from electric_multipoles import ChargeDistribution, e
import numpy as np

a_array = [ 1e-10, 5e-10, 24e-10]   # m

r_o = np.array([0, 0, 0]) * 1e-10

chargeArray = [e]*4 + [2*e]*2 + [-8*e]

for a in a_array :

    chargePosArray = [ 
        [ a, 0, 0],
        [ 0, a, 0],
        [-a, 0, 0],
        [ 0,-a, 0],
        [ 0, 0, a],
        [ 0, 0,-a],
        [ 0, 0, 0]
    ]

    myChargeDist = ChargeDistribution(chargeArray, chargePosArray)

    print('\na = ', a, ' m')

    print('Q_e : ')
    print(myChargeDist.Q_e(r_o))