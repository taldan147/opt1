import numpy as np
from numpy import linalg as LA

def GramShmidthQR(A):
    n=len(A[0])
    R = np.zeros((3,3), dtype=int)
    R[0][0] = LA.norm(A[:,0])
    q1 = A[:,0]/R[0][0]
    Q = [q1]
    for i in range (1,n-1):
        qi = A[:,i]
        for j in range (0,i-1):
            R[j][i] = Q[j]*qi
            qi = qi- R[j][i]*Q[j]
        R[i][i] = LA.norm(qi)
        qi = qi/R[i][i]
        Q.append(qi)
    return R,Q

A = np.asarray([[-1,-1,1], [1,3,3], [-1,-1,5], [1,3,7]])
R,Q = GramShmidthQR(A)
print(R)
print(Q)
