from VHE import *
from VHE_new import *
import numpy as np
import random
N = 10
    #x1 = np.array([1,2,3,4])
    #x2= np.array([1,2,3,4])
    
x1 = np.random.randint(-2, 2, (N))
print("x1 is")
print(x1)
x2 = np.random.randint(-500, 500, (N))
print("x2 is")
print(x1)
    
T = getrandomMatrix(N, N, t_bound)

S = getSecretKey(T)
c1 = encrypt(T, x1)
c2 = encrypt(T, x2)
c3 = c1 + c2
p1 = decrypt(S, c1)
p3 = decrypt(S, c3)

    
print(c1)


print(decrypt(S, c1))
print("input x1", x1)
print(c1)
print(p1)
print("input x2", x2)
print(x1 + x2)
print(decrypt(S, c1 + c2))
    
M = innerProdClint(T)
print(c1)
cc = innerProd(c1, c1, M)
dxx = decrypt(S, cc)
xx = np.dot(x1, x1)

print("xx", xx)
print("dxx", dxx[0])
print("dot product difference", xx - dxx[0])
#x1=np.transpose(x1)
#print(x1)
#print(x2)

#x3=np.outer(x1,x2)
#print(x3)
"""
T=getrandomMatrix(N,N,10000)
#print(T)
S=getSecretKey(T)
#print(S.shape)
#print(getSecretKey(T))
c1=encrypt(T,x1)
c2=encrypt(T,x1)
print("c1",c1)
c3=decrypt(S,c2)
print("c3",c3)
#s=np.ones([10,10])
#print(getBitMatrix(T))
c=addVector(c1,c2)
print(c)
F=decrypt(S,c)
print(F)
"""


