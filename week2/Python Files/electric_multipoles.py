import numpy as np
from ai.cs import sp2cart
from itertools import combinations

e   =   1.602176487e-19    # C
h   =   6.62607015e-34     # J - s
ep0 =   8.8541878128e-12   # F - m

one_over_4_pi_e0    =   1./ ( 4 * np.pi * ep0 )     # N - m^2 / C^2

# Electric field E at a position r
# due to a charge Q at position rQ
def E(Q, rQ, r) :

    if (( r - rQ ) == 0 ).all() :
        
        return np.array( [ np.inf, np.inf, np.inf ] )

    else :

        return ( one_over_4_pi_e0 * Q / ( np.linalg.norm( r - rQ )**3 ) ) * (r - rQ)   # N / C

# Potential V at a position r
# due to a charge Q at position rQ
# assuming V at r->inf is 0
def V(Q, rQ, r) :

    if (( r - rQ ) == 0 ).all() :

        return np.inf

    else :

        return one_over_4_pi_e0 * Q / np.linalg.norm( r - rQ )


def partial_derivative(f, x_vector, j) :

    x_j = x_vector[j]

    epsilon = 1e-20
    
    if x_j > 1e-10 :
        
        epsilon *= x_j

    x_vector_2 = np.copy(x_vector)

    x_vector_2[j] += epsilon

    return ( f(x_vector_2) - f(x_vector) ) / epsilon

def grad(f, r) :

    return np.array([ partial_derivative(f, r, i) for i in range(3) ])

# Charge C
# has a Electric charge Q
# and a position vector r
class Charge() :

    # Constructor
    def __init__(self, Q, x, y, z) :

        self.Q = Q
        self.r = np.array([x, y, z])

    # Electric field E due to C
    # at a position whose position vector is r
    def E(self, r) :

        return E(self.Q, self.r, r)

    # Potential V due to C
    # at a position whose position vector is r
    def V(self, r) :

        return V(self.Q, self.r, r)

    # Potential energy PE due to 
    # potential V at the position of C
    def PE(self, V) :

        return self.Q * V(self.r)

    # Dipole moment P_e about origin
    # due to C
    def P_e(self, origin=np.zeros(3)) :

        return self.Q * (self.r - origin)

    # Second moment q_e about origin
    # due to C
    def q_e(self, origin=np.zeros(3)) :

        return np.outer(self.P_e(origin) , (self.r - origin) )

    # Quadrupole moment Q_e about origin
    # due to C
    def Q_e(self, origin=np.zeros(3)) :

        return 3 * self.q_e(origin) - self.Q * ( np.linalg.norm(self.r - origin) ** 2 ) * np.identity(3)

# ChargeDistribution chargeDistro
# has a list of charges - Charges
class ChargeDistribution() :

    # Constructor
    def __init__(self, Q_array, Coordinates_array) :

        if len(Q_array) != len(Coordinates_array) :

            raise RuntimeError('Length of arrays do not match')

        self.Charges = [ Charge(Q, r[0], r[1], r[2]) for Q, r in zip(Q_array, Coordinates_array) ]

    # Mean position of all charges
    # in chargeDistro
    def meanPosition(self) :

        return np.mean( [ C.r for C in self.Charges ] , axis=0 )

    # Electric field at position r due to chargeDistro
    # is equal to summation of E due to all charges
    def E(self, r) :

        return np.sum( np.array([ C.E(r) for C in self.Charges ]) , axis=0 )

    # Force on charge Charges[ chargeIndex ]
    # due to other charges in the chargeDistro
    def F_internal(self, chargeIndex) :

        r = self.Charges[chargeIndex].r

        return np.sum( [ C.E(r) for C in ( self.Charges[ : chargeIndex ] + self.Charges[ (chargeIndex+1) : ] ) ] , axis=0 ) * self.Charges[chargeIndex].Q

    # Potential at a position r
    # due to chargeDistro
    def V(self, r) :

        return np.sum([ C.V(r) for C in self.Charges ])

    # Potential Energy due to interaction
    # of charges within chargeDistro
    def PE_internal(self) :

        PE = 0.

        for i,j in combinations(range(len(self.Charges)), 2) :

            PE += self.Charges[j].PE(self.Charges[i].V)

        return PE

    # Potential energy of the chargeDistro
    # due to a potential field V (callable)
    def PE_external(self, V) :

        return np.sum( [C.PE(V) for C in self.Charges] , axis=0 )

    # Dipole moment of the chargeDistro
    # about the point origin
    def P_e(self, origin=np.zeros(3)) :

        return np.sum( [ C.P_e(origin) for C in self.Charges ], axis=0 )

    # Second moment of the chargeDistro
    # about the point origin
    def q_e(self, origin=np.zeros(3)) :

        return np.sum( [ C.q_e(origin) for C in self.Charges ], axis=0 )

    # Quadrupole moment of the chargeDistro
    # about the point origin
    def Q_e(self, origin=np.zeros(3)) :

        return np.sum( [ C.Q_e(origin) for C in self.Charges ], axis=0 )

# Dipole DP
# has a dipole moment vector p
# and a position vector r
class Dipole() :

    # Constructor
    def __init__(self, p, x, y, z) :

        self.p = np.array(p)
        self.r = np.array([x,y,z])

    # Constructor to create a dipole
    # from a ChargeDistribution object
    # The object will be treated as a pure dipole
    # with r equal to the mean position
    # of the charge distribution
    @classmethod
    def fromChargeDistro(cls, chargeDistro) :

        r = chargeDistro.meanPosition()

        return cls(chargeDistro.P_e(), r[0], r[1], r[2])  

    # Electric field E due to DP
    # at a point with position vector r
    def E(self, r) :

        mod_r = np.linalg.norm(r)

        J_r_r3 = np.identity(3) * ( 1 / mod_r**3) - 3 * np.outer(r,r) * (1 / mod_r**5) 

        return - one_over_4_pi_e0 * np.dot(self.P_e, J_r_r3)

    # Potential V due to DP
    # at a position whose position vector is r
    def V(self, r) :

        delta_r = r - self.r

        return one_over_4_pi_e0 * np.inner(self.p, delta_r) / ( np.linalg.norm(delta_r) ** 3 )

    # Dipole moment P_e
    def P_e(self) :

        return self.p

    # Potential energy PE due to 
    # potential V at the position of DP
    def PE(self, V) :

        return np.inner( self.P_e(), grad(V, self.r) )

    # Rotate and set the direction of p
    # according to physics spherical coordinates notation
    # https://en.wikipedia.org/wiki/Spherical_coordinate_system#/media/File:3D_Spherical.svg
    def set_theta_phi(self, theta, phi) :

        self.p = sp2cart(np.linalg.norm(self.p), theta, phi)
        pass

    



