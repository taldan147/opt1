import math
from functools import reduce

import numpy as np
from numpy import linalg as LA

def GramShmidthQR(A):
    n=len(A[0])
    R = np.zeros((n,n))
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
    return R,np.asarray(Q).transpose()

def ModifiedGramShmidthQR(A):
    n=len(A[0])
    R = np.zeros((n,n))
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
    return R,np.asarray(Q).transpose()

epsilon = 1
A = np.asarray([[1,1,1], [epsilon, 0,0], [0,epsilon,0], [0,0,epsilon]])
R1,Q1 = GramShmidthQR(A)
print("R1", R1)
print("Q1", Q1)

epsilon = 10**-10
A = np.asarray([[1,1,1], [epsilon, 0,0], [0,epsilon,0], [0,0,epsilon]])
R2,Q2 = GramShmidthQR(A)
print("R2",R2)
print("Q2",Q2)

epsilon = 1
A = np.asarray([[1,1,1], [epsilon, 0,0], [0,epsilon,0], [0,0,epsilon]])
R3,Q3 = ModifiedGramShmidthQR(A)
print("R3",R3)
print("Q3",Q3)

epsilon = 10**-10
A = np.asarray([[1,1,1], [epsilon, 0,0], [0,epsilon,0], [0,0,epsilon]])
R4,Q4 = ModifiedGramShmidthQR(A)
print("R4",R4)
print("Q4",Q4)

# ----------------------------------------------- c --------------------------------------------------

norm1 = LA.norm((Q1.transpose()@Q1)-np.eye(3))
print(norm1)

norm2 = LA.norm((Q2.transpose()@Q2)-np.eye(3))
print(norm2)

norm3 = LA.norm((Q3.transpose()@Q3)-np.eye(3))
print(norm3)

norm4 = LA.norm((Q4.transpose()@Q4)-np.eye(3))
print(norm4)



