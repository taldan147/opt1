import numpy as np
from numpy import linalg as LA

A = np.asarray([[1,2,3,4], [2,4,-4,8], [-5,4,1,5], [5,0,-3,-7]])
ATA = (A.transpose()@A)

eig = LA.eig(ATA)

indexOfMaxEig = np.argmax(eig[0])

v=np.asarray([[eig[1][0][indexOfMaxEig]],[eig[1][1][indexOfMaxEig]],[eig[1][2][indexOfMaxEig]],[eig[1][3][indexOfMaxEig]]])

print("v",v)

# calculate L2 norm of A
AVl2 = LA.norm(A@v)
vl2 = LA.norm(v)
Al2 = AVl2/vl2

print("AV", Al2)



print(np.triu(A))

