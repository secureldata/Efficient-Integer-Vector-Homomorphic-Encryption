import numpy as np
import random
N=4
c1 = np.random.randint(-2, 2, (N))
print(c1)
c2 = np.random.randint(-500, 500, (N))
print(c2)
c1 = np.expand_dims(c1, 0)#np.expand_dims:用于扩展数组的形状在第一个维度扩展形状
print(c1)
c2 = np.expand_dims(c2, 0)
print(c2)
c2 = c2.T
print(c2)
print(np.multiply(c1,c2))
cc = np.multiply(c1,c2).flatten()
print(cc)
