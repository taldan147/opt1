from numpy import linalg as LA
import numpy as np

A = np.asarray([[2,1,2], [1,-2,1], [1,2,3], [1,1,1]])
AT=A.transpose()
ATA = (AT@A)
print("ATA",ATA)
b = np.asarray([[6],[1],[5],[2]])
# ----------------------------------------------- a --------------------------------------------------
L = LA.cholesky(ATA)
print("L", L)
LLT = (L@L.transpose())
ATb = (AT@b)
print("ATb",ATb)
LLTinv = LA.inv(LLT)
x = LLTinv@ATb
print("cholesky x" ,x)
# ----------------------------------------------- b --------------------------------------------------

# x=R^-1Q^Tb QR factorization

q, r = np.linalg.qr(A)
print("Q",q)
print("R",r)

x2 = LA.inv(r)@q.transpose()@b
print("QR x", x2)

# SVD facorization x = Vy = Vsimga^-1U^Tb

(U,S,V) = LA.svd(A, full_matrices=False)
print("U",U)
print("S", S)
V=V.transpose()
print("V",V)

Utb = (U.transpose())@b

y = np.asarray([Utb[0]/S[0], Utb[1]/S[1], Utb[2]/S[2]])
print("y",y)
x3 = V@y
print("SVD x",x3)

# ----------------------------------------------- 4c --------------------------------------------------

r = (A@x) - b
print("r", r)
Atr = AT@r
print("ATr", Atr)

# ----------------------------------------------- 4d --------------------------------------------------

W = np.asarray([[808,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
ATWA = AT@W@A
ATWAinv = LA.inv(ATWA)
xw = ATWAinv@AT@W@b
rw = (A@xw)-b
print("r w",rw)