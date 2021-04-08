from numpy import linalg as LA
import numpy as np

A = np.asarray([[2,1,2], [1,-2,1], [1,2,3], [1,1,1]])
AT=A.transpose()
ATA = (AT@A)

b = np.asarray([[6],[1],[5],[2]])


# ----------------------------------------------- a --------------------------------------------------
L = LA.cholesky(ATA)
LT = L.transpose()

# A^T A=LL^T,x=(A^T A)^(-1) A^T b⇒ x=(LL^T )^(-1) A^Tb

LLT = (L@LT)

ATb = (AT@b)

LLTinv = LA.inv(LLT)

x = LLTinv@ATb

print(x)

# ----------------------------------------------- b --------------------------------------------------

# x=R^-1Q^Tb QR factorization

q, r = np.linalg.qr(A)

x2 = LA.inv(r)@q.transpose()@b
print(x2)

# SVD facorization x = Vy = Vsimga^-1U^Tb



(U,S,V) = LA.svd(A, full_matrices=False)

print("U",U)
print("S", S)
V=V.transpose()
print("V",V)


Utb = (U.transpose())@b
print("utb" , Utb)





# S= np.asarray([[S[0]],[S[1]],[S[2]]])

y = np.asarray([Utb[0]/S[0], Utb[1]/S[1], Utb[2]/S[2]])
# y = Utb1/S
print("y",y)

x3 = V@y
print(x3)


# ----------------------------------------------- 4c --------------------------------------------------

r = (A@x) - b

print("r", r)

Atr = AT@r

print("ATr", Atr)