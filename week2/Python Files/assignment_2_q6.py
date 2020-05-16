from electric_multipoles import ChargeDistribution, Dipole, e
from ai.cs import sp2cart
from math import asin, pi
import numpy as np


# Radius of Carbon atom
r_C = 70e-12 # m

# Radius of Hydrogen atom
r_H = 53e-12 # m

# Carbon - Hydrogen Bond Length
BL_CH = 160e-12 # m

# H atom PE
PE_H = 2 * -2.179871971e-18 #j

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

def potentialDue2Hatoms(r) :

    return H1.V(r) + H2.V(r) + H3.V(r) + H4.V(r)

PE = C_ChargeDist.PE_internal() + C_ChargeDist.PE_external(potentialDue2Hatoms)
PE += H1.PE(H2.V) + H1.PE(H3.V) + H1.PE(H4.V)
PE += H2.PE(H3.V) + H2.PE(H4.V)
PE += H3.PE(H4.V)
PE += 4*PE_H

print('\nTotal Potential Energy of CH4 : ')
print(PE, ' J')
print(PE/e, ' eV')