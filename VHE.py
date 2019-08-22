import numpy as np
import random  
l=100
l = 100
w = 2**45
a_bound, e_bound, t_bound = 100, 1000, 100
def getrandomMatrix(row,col,tbound):#bound在C++中表示大整数，如何在python中表示大整数呢？
    A = np.arange(row*col)
    A = A.reshape((row, col))
    for i in range(row):
        for j in range(col):
            A[i][j]=random.randint(0, tbound)
    return A
def getBitMatrix(S):# 大整数矩阵S 应该是numpy矩阵，全部都用此类矩阵
    row,col=S.shape
    results = np.ones([row,l*col])
    #print(results.shape)
    powers = np.ones(l,dtype=int) 
    powers[0]=l
    for i in range(l-1):
        powers[i]=powers[i]*2
    for i in range(row):
        for j in range(col):
            for k in range(l):
                results[i][j*l+k]=S[i][j]*powers[k]
    #print(results.shape)
    return results
def getBitVector(c):#此处c应该是为向量
    length=len(c)
    result=np.ones(length*l,dtype=int)
    for i in range(length):
        if c[i]<0:
            sign=-1
        else:
            sign=1
        value=sign*c[i]
        c1=intTo2Str(value,l)
        for j in range(l):
            result[i*l+j]=sign*c1[j]
        return result
def intTo2Str( X , K ):#X为数字，#K为输出的二进制位数
    X=int(X)
    a=bin(X).replace('0b','')
    a=str(a)
    result=np.zeros(K)
    print(a)
    l=len(a)
    for i in range(l):
        result[i]=int(a[i])
    return result
def getSecretKey(T):#输入矩阵T
    row,col=T.shape
    #print(row)
    #print(col)
    I = np.eye(row)
    #print(I)
    return hCat(I,T)
def hCat(A,B):#组合row行数相同的两个矩阵
    result=np.hstack((A,B))
    return result
def vCat(A,B):#组合col列上相同的两耳矩阵
    result=np.vstack((A,B))
    return  result

def nearest_integer(x):
    global w
    return int((x + (w + 1) / 2) / w)


def decrypt(S, c):
    sc = np.dot(S, c)
    nearest_int_vec = np.vectorize(nearest_integer) 
    """
    numpy.vectorize(pyfunc, otypes=None, doc=None, excluded=None, cache=False, signature=None)
    Parameters:	
    pyfunc :python函数或方法
    otypes : 输出数据类型。必须将其指定为一个typecode字符串或一个数据类型说明符列表。每个输出应该有一个数据类型说明符。
    doc : 函数的docstring。如果为None，则docstring将是 pyfunc.__doc__。
    excluded : 表示函数不会向量化的位置或关键字参数的字符串或整数集。这些将直接传递给未经修改的pyfunc
    cache :如果为True，则缓存第一个函数调用，该函数调用确定未提供otype的输出数。
    signature : 广义通用函数签名，例如，(m,n),(n)->(m)用于矢量化矩阵 - 向量乘法。如果提供的话，pyfunc将调用（并期望返回）具有由相应核心维度的大小给出的形状的数组。默认情况下，pyfunc假定将标量作为输入和输出。
    Returns:	
    vectorized :向量化的数组
    """
    negative_vec = np.vectorize(lambda x: x-1 if x < 0 else x)
    return negative_vec(nearest_int_vec(sc))
def keySwitchMatrix(S,T): #S,T同为矩阵
    start=getBitMatrix(S)
    #print(start.shape)
    A=getrandomMatrix(len(T),len(start[0]),1000)
    #print(A.shape)
    E=getrandomMatrix(len(start),len(start[0]),1000)
    #print(E.shape)
    #print(T.shape)
    F = np.matmul(T,A)
    return vCat(start+E-F,A)
def encrypt(T,x):#T为公开密钥矩阵 x为原始数据向量
    l1=len(x)
    result=np.eye([l1,l1],dtype=int)
    #print(result)
    return keySwitch(keySwitchMatrix(result,T),w*x)
def keySwitch(M,c):#M为密钥转换矩阵 c为密文
    cstar=getBitVector(c)
    return np.dot(M,cstar)
def addVector(c1,c2):#密文向量C1 和C2
    return c1+c2
def linerTransform(M,c): #矩阵M,密文c
    return M*getBitVector(c)
def innerTransformClirnt(G,S,T):#矩阵G，S，T
    return keySwitchMatrix(np.dot(G, S),T)
def innerProd(c1,c2,M):#密文向量c1 和c2 
    c1 = np.expand_dims(c1, 0)#np.expand_dims:用于扩展数组的形状在第一个维度扩展形状
    c2 = np.expand_dims(c2, 0)
    c2 = c2.T
    cc = np.multiply(c1,c2).flatten()
    """
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
    c1=[-1  1  1 -2]
    c2=[ -57   23 -477 -331]
    c1扩展第一维=[[-1  1  1 -2]]
    c2扩展第一维=[[ -57   23 -477 -331]]
    c2转置=[[ -57]
            [  23]
            [-477]
            [-331]]
    c1*c2=
    [[  57  -57  -57  114]
    [ -23   23   23  -46]
    [ 477 -477 -477  954]
    [ 331 -331 -331  662]]
    向量一维化
    [  57  -57  -57  114  -23   23   23  -46  477 -477 -477  954  331 -331
    -331  662]
    """
    #flatten是numpy.ndarray.flatten的一个函数，即返回一个一维数组。
    #flatten只能适用于numpy对象，即array或者mat，普通的list列表不适用！。
    nearest_int_vec = np.vectorize(nearest_integer, otypes=[np.object])
    cc = nearest_int_vec(cc)
    return np.dot(M, getBitVector(cc))

def innerProdClint(T):#矩阵T
    S = getSecretKey(T)
    tvsts = (np.matmul(S.T, S)).flatten().T
    tvsts = np.expand_dims(tvsts, 0)
    mvsts = np.repeat(tvsts, T.shape[0], axis=0)
    return keySwitchMatrix(mvsts, T)
"""
def copyRows(row, numrows):#其中row为矩阵，numrows为long整形
    ans=np.ones([numrows,len(row[0])])
    for i in range(len(row)):
        for j in range(len(row[0])):
            ans[i][j]=row[0][j]
    return ans
def vectorize(M):
    ans=np.ones([len(M)*len(M[0])],l)
    for i in range(len(M)):
        for j in range(len(M[0])):
            ans[i*len(M[0])+j]=M[i][j]
    return ans
"""


