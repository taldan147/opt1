import math
from functools import reduce

import numpy as np
from numpy import linalg as LA

def GramShmidthQR(A):
    n=len(A[0])
    R = np.zeros((n,n), dtype=int)
    R[0][0] = LA.norm(A[:,0])
    q1 = A[:,0]/R[0][0]
    Q = [q1]
    for i in range(1,n):
        qi = A[:,i]
        for j in range(i):
            R[j][i] = reduce(lambda x, y: x + y, Q[j]*A[:,i])
            qi = qi- R[j][i]*Q[j]
        R[i][i] = LA.norm(qi)
        qi = qi/R[i][i]
        Q.append(qi)
    return R,np.asarray(Q, dtype=int).transpose()

def ModifiedGramShmidthQR(A):
    n=len(A[0])
    R = np.zeros((n,n), dtype=int)
    R[0][0] = LA.norm(A[:,0])
    q1 = A[:,0]/R[0][0]
    Q = [q1]
    for i in range(1,n):
        qi = A[:,i]
        for j in range(i):
            R[j][i] = reduce(lambda x, y: x + y, Q[j]*qi)
            qi = qi- R[j][i]*Q[j]
        R[i][i] = LA.norm(qi)
        qi = qi/R[i][i]
        Q.append(qi)
    return R,Q

# epsilon = 1
# A = np.asarray([[1,1,1], [epsilon, 0,0], [0,epsilon,0], [0,0,epsilon]])
# R,Q = GramShmidthQR(A)
# print("e=1")
# print(R)
# print(Q)

epsilon = math.e**-10
A = np.asarray([[1,1,1], [epsilon, 0,0], [0,epsilon,0], [0,0,epsilon]])
print(A)
# R,Q = GramShmidthQR(A)
# print("1e-10")
# print(R)
# print(Q)
