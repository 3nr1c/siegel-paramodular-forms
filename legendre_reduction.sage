def legendre_reduce(T, N=277):
    U = matrix.identity(2)
    v = vector([0,1])
    S = Matrix([[0,-1], [1, 0]])

    while not (2 * abs(T[0][1]) <= T[0][0] <= T[1][1]):
        if T[1][1] < T[0][0]:
            v = S.inverse() * v
            U = U * S
            T = S.T * T * S
    
        if 2 * abs(T[0][1]) > T[0][0]:
            L = floor(-T[0][1]/T[0][0] + 1/2)
            R = Matrix([[1, L], [0, 1]])
            
            v = R.inverse() * v
            U = U * R
            T = R.T * T * R
    
    if T[0][1] < 0:
        R = Matrix([[1, 0], [0, -1]])
        
        v = R.inverse() * v
        U = U * R
        T = R.T * T * R

    return T, U, v

def aut_T_equivalent(T, v1, v2, N=277):
    q = QuadraticForm(ZZ, 2, [T[0][0], 2*T[1][0], T[1][1]])
    for aut in q.automorphisms():
        autv1 = aut * v1
        if (autv1[0] * v2[1] - autv1[1] * v2[0]) % N == 0:
            return True
    return False
