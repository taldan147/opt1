import numpy as np
from numpy import linalg as LA

A = np.asarray([[1,2,3,4], [2,4,-4,8], [-5,4,1,5], [5,0,-3,-7]])
ATA = (A.transpose()@A)

l2norm = LA.norm(A,2)

eig = LA.eig(ATA)

indexOfMaxEig = np.argmax(eig[0])

print("eig", eig)
print("index", indexOfMaxEig)

v=np.asarray([[eig[1][0][indexOfMaxEig]],[eig[1][1][indexOfMaxEig]],[eig[1][2][indexOfMaxEig]],[eig[1][3][indexOfMaxEig]]])

AV= (A@v)

ATAv= (ATA@v)

print("ATAV", ATAv)

print("192", v*192)

AVl2 = LA.norm(AV)

vl2 = LA.norm(v)

Al2 = AVl2/vl2

print("l2 nortm" ,l2norm)


print("AV", Al2)