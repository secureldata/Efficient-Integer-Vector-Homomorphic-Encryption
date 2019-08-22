import numpy as np
l = 100
w = 2**45
a_bound, e_bound, t_bound = 100, 1000, 100
# finds c* then returns Mc*
def key_switch(M, c):
    cstar = get_bit_vector(c)
    return np.dot(M, cstar)
def get_random_matrix(row, col, bound):
    return np.random.randint(0, bound, (row, col))
# returns S*
def get_bit_matrix(S):
    global l
    powers = np.array([2**x for x in range(l)])
    result = np.array([np.array([powers * int(elem) for elem in row]) for row in S])
    result = result.reshape(result.shape[0], result.shape[1] * result.shape[2])
    return result
# returns c*
def get_bit_vector(c):
    global l

    result = np.array([bin(elem).replace('b', '').zfill(l)[::-1] for elem in c])

    ans = np.zeros((l*len(c)))

    for i, string in enumerate(result):
        string = list(string)
        negate = 1
        if string[-1] == "-":
            string[-1] = '0'
            negate = -1
        for j, char in enumerate(string):
            ans[i*l + j] = int(char) * negate

    return ans
# returns S
def get_secret_key(T):
    rows, cols = T.shape
    I = np.eye(rows)
    return np.hstack((I, T))

def nearest_integer(x):
    global w
    return int((x + (w + 1) / 2) / w)

def decrypt(S, c):
    sc = np.dot(S, c)
    nearest_int_vec = np.vectorize(nearest_integer)
    negative_vec = np.vectorize(lambda x: x-1 if x < 0 else x)
    return negative_vec(nearest_int_vec(sc))

def key_switch_matrix(S, T):
    t_rows, t_cols = T.shape
    s_star = get_bit_matrix(S)
    s_rows, s_cols = s_star.shape
    A = get_random_matrix(t_cols, s_cols, a_bound)
    E = get_random_matrix(s_rows, s_cols, e_bound)
    ans = np.vstack((s_star + E - np.matmul(T, A), A))
    return ans

def encrypt(T, x):
    I = np.eye(len(x), dtype=int)
    return key_switch(key_switch_matrix(I, T), w * x)

def linear_transform(M, c):
    return M * get_bit_vector(c)

def linear_transform_client(G, S, T):
    return key_switch_matrix(np.dot(G, S), T)

def inner_prod(c1, c2, M):
    c1 = np.expand_dims(c1, 0)#np.expand_dims:用于扩展数组的形状在第一个维度扩展形状
    c2 = np.expand_dims(c2, 0)
    c2 = c2.T
    cc = np.multiply(c1,c2).flatten()

    nearest_int_vec = np.vectorize(nearest_integer, otypes=[np.object])
    cc = nearest_int_vec(cc)
    
    return np.dot(M, get_bit_vector(cc))


def inner_prod_client(T):
    S = get_secret_key(T)
    tvsts = (np.matmul(S.T, S)).flatten().T
    tvsts = np.expand_dims(tvsts, 0)
    mvsts = np.repeat(tvsts, T.shape[0], axis=0)
    return key_switch_matrix(mvsts, T)


if __name__ == "__main__":
    N = 10
    #x1 = np.array([1,2,3,4])
    #x2= np.array([1,2,3,4])
    
    x1 = np.random.randint(-2, 2, (N))
    print("x1 is")
    print(x1)
    x2 = np.random.randint(-500, 500, (N))
    print("x2 is")
    print(x1)
    
    T = get_random_matrix(N, N, t_bound)

    S = get_secret_key(T)
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
    
    M = inner_prod_client(T)
    print(c1)
    cc = inner_prod(c1, c1, M)
    dxx = decrypt(S, cc)
    xx = np.dot(x1, x1)

    print("xx", xx)
    print("dxx", dxx[0])
    print("dot product difference", xx - dxx[0])

    def activation_hack(x):
        M = inner_prod_client(T)
        x = encrypt(T, np.array([x], dtype=np.int64))
        y = inner_prod(x, x, M)
        return y

    #square = np.vectorize(activation_hack)
    #print(decrypt(S, c1))
    #print(decrypt(S, square(c1)))
