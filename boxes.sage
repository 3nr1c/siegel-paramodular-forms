import numpy as np
load("legendre_reduction.sage")

def S_box(N, G, u, d):
    triples = []

    a,b,c = G[0][0], G[0][1], G[1][1]
    Delta = det(G)
    # print(f"∆ = {Delta}")
    
    m_upper_bound = a * (u + sqrt(u^2-d*Delta)) / (2 * Delta * N)

    M = np.arange(1, floor(m_upper_bound) + 1, dtype="int")
    X = (4*a*u*N)*M - a*a * d - (4*Delta) * np.square(M*N)

    M = M[X >= 0]
    X = X[X >= 0]

    for m, x in zip(M, X):
        r_lower_bound = (-2 * b * m * N - sqrt(x)) / a
        r_upper_bound = (-2 * b * m * N + sqrt(x)) / a

        R = np.arange(ceil(r_lower_bound), floor(r_upper_bound) + 1, 1, dtype="int")
        N_lower = (np.square(R) + d)/(4 * m * N)
        N_upper = (u - b*R - c*m*N) / a

        mask = N_lower < N_upper
        R = R[mask]
        N_lower = N_lower[mask]
        N_upper = N_upper[mask]
        for r, n_lower, n_upper in zip(R, N_lower, N_upper):
            for n in np.arange(ceil(n_lower), floor(n_upper) + 1, 1, dtype="int"):
                yield (
                    Integer(4*N*m*n-r^2),
                    Matrix([[Integer(n),Integer(r)/2],[Integer(r)/2,Integer(N*m)]])
                )


    for r in range(-d, d):
        if r^2 > -d:
            continue

        n_upper_bound = (u - b*r) / a
        for n in range(1, floor(n_upper_bound) + 1):
            #triples.append((0,r,n))
            yield (
                -r^2,
                Matrix([[n,r/2],[r/2,0]])
            )

    for r in range(-d, 0):
        if r^2 <= -d:
            #triples.append((0,r,0))
            yield (
                -r^2,
                Matrix([[0,r/2],[r/2,0]])
            )

    # return triples

def det_box(max_det, N):
    for D in range(1, max_det+1):
        queue = []
        done = []

        if kronecker(-D, N) == -1:
            continue
        r1 = ZZ.quotient(N)(-D).sqrt()
        r2 = -r1

        for b in range(5):
            r = r1.lift() + b*N
            a = 0
            while D + r^2 > 4*a*N:
                a += 1
            if D+r^2 == 4*a*N:
                queue.append([r, a])

        for b in range(5):
            r = r2.lift() + b*N
            a = 0
            while D + r^2 > 4*a*N:
                a += 1
            if D+r^2 == 4*a*N:
                queue.append([r, a])

        for r, a in queue:
            for n in a.divisors():
                m = a/n
                # print(n,r,m*N," ",4*n*m*N-r^2,D)

                mat = Matrix([[n, r/2],[r/2,N*m]])

                found_before = False
                for nat in done:
                    T, _, v = legendre_reduce(nat, N)
                    S, _, w = legendre_reduce(mat, N)
                    if T == S and aut_T_equivalent(T, v, w, N):
                        found_before = True
                        break

                if not found_before:
                    done.append(mat)
                    yield mat

    return

# 0<=b<=a<=c 
def abc_box(max_c, max_det=-1, N=277):
    R.<x> = ZZ.quotient(N)[]
    done = []
    
    for c in tqdm(range(1, max_c+1)):
        for a in range(0, min(c, max_det//(4*c)) + 1):
            lower = 4*a*c-max_det if max_det > 0 else 0
            for b in range(floor(sqrt(max(0, lower))), min(a+1,ceil(float(sqrt(4*a*c))))):
                # we have a tuple a,b,c
                TT = Matrix([
                    [a, b/2],
                    [b/2, c]
                ])

                f = a*x^2 + b*x + c

                for v in f.roots(multiplicities=False):
                    U = Matrix([
                        [1, v.lift()],
                        [0, 1]
                    ])
                    T = U.T * TT * U
                    assert T[1][1] % N == 0

                    found_before = False
                    for mat in done:
                        R, _, v = legendre_reduce(mat, N)
                        S, _, w = legendre_reduce(T, N)
                        if R == S and aut_T_equivalent(R, v, w, N):
                            found_before = True
                            break

                    if not found_before:
                        done.append(T)
                        yield T


                U = Matrix([
                    [0,  1],
                    [-1, 0]
                ])
                T = U.T * TT * U

                if T[1][1] % N == 0:
                    found_before = False
                    for mat in done:
                        R, _, v = legendre_reduce(mat, N)
                        S, _, w = legendre_reduce(T, N)
                        if R == S and aut_T_equivalent(R, v, w, N):
                            found_before = True
                            break

                    if not found_before:
                        done.append(T)
                        yield T

"""
p = 3
N = 277

G = 2*Matrix([
    [1,     233/2],
    [233/2, 13573]
]).adjugate() / p

a, b, c = G[0][0], G[1][0], G[1][1] * N

""G = Matrix([
    [a / p,     b],
    [b,     p*c/N]
])""

print(G)

A = set(S(N, G, 15, 0))
B = set(S(N, G, 14, 0))

print(A.difference(B))"""