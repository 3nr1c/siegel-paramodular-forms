import numpy as np

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