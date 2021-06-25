import json
import re
from collections import defaultdict
from tqdm import tqdm
from functools import partial
from collections.abc import Iterable
import numpy as np
import multiprocessing as mp
from itertools import product
from scipy.interpolate import approximate_taylor_polynomial
load("boxes.sage")
load("legendre_reduction.sage")

def read_coefficients(file):
    with open(file) as json_file:
        data = json.load(json_file)
    
    coefficients = defaultdict(list)

    for determinant, values in data["Fourier_coefficients"].items():
        determinant = int(determinant)

        for matrix_str, coefficient in values.items():
            parsed = re.search(r'\((-?\d+), (-?\d+), (-?\d+)\)', matrix_str)
            n, r, Nm = int(parsed.group(3)), int(parsed.group(2)), int(parsed.group(1))
            coefficients[determinant].append([
                tuple2matrix((Nm, r, n)), QQ(coefficient)
            ])

    return coefficients


def specialization_q_expansion(f, N, s, t=matrix.zero(2), max_n=20, boxes=None, warnings=True):
    q_expansion = defaultdict(int)
    F = CyclotomicField(t.denominator())
    S.<q> = PuiseuxSeriesRing(F, sparse=True)

    q_expansion = S(0)

    if boxes is None:
        boxes = list(S_box(N,s,max_n,0))
    # print(boxes)

    for detT, T in boxes:
        e = (s * T).trace()

        T_legendre, _, v = legendre_reduce(T, N)
        if e <= max_n:
            found_flag = False
            for U, aT in f[detT]:
                U_legendre, _, w = legendre_reduce(U, N)
                """if U_legendre == T_legendre:
                    print(U_legendre)
                    print(U)
                    print(T)
                    print(v,w)"""
                if U_legendre == T_legendre and aut_T_equivalent(U_legendre, v, w):
                    found_flag = True
                    Tt = (T * t).trace()
                    q_expansion += F(exp(2*pi*I*Tt) * aT) * q^e
                    break
            if not found_flag:
                print(f"Warning - might be missing for trace {e}: T = {(T[1][1],2*T[1][0],T[0][0])}, det(T) = {detT}")
                # if e.is_integer():
                    # raise

    return q_expansion.add_bigoh(max_n+1)


def Hecke_Tp_q(f, s, p, N=277, max_n=20, warnings=None):
    a, b, c = s[0][0], s[0][1], s[1][1]*N
    try:
        G = p*s
        Tpf = p * specialization_q_expansion(f, N, G, max_n=max_n)
        
        G = Matrix([[a/p, b], [b,p*c/N]])
        if a % p != 0:
            f_i = specialization_q_expansion(f, N,
                G,
                max_n=max_n, warnings=warnings)
            Tpf += f_i
        else:
            G_box = list(S_box(N, G, max_n, 0))
            for i in range(p):
                f_i = specialization_q_expansion(f, N,
                    G,
                    t=Matrix([[i/p,0],[0,0]]),
                    max_n=max_n,
                    boxes=G_box)
                Tpf += f_i / p

        ##########################

        for i in range(p):
            G = Matrix([[p*a, b+i*a], [b+i*a, (c/N + 2*i*b + i^2*a)/p]])
            G_box = list(S_box(N, G, max_n, 0))

            if (c + 2*i*b*N + i^2*a*N) % p != 0:
                f_i = specialization_q_expansion(f, N,
                        G,
                        max_n=max_n,
                        boxes=G_box)
                Tpf += f_i
            else:
                for j in range(p):
                    f_ij = specialization_q_expansion(f, N,
                        G,
                        t=Matrix([[0,0],[0,j/p]]),
                        max_n=max_n,
                        boxes=G_box)
                    Tpf += f_ij / p

        ###########################


        G = s/p
        G_box = list(S_box(N, G, max_n, 0))
        if a % p != 0:
            for j in range(p):
                for k in range(p):
                    f_jk = specialization_q_expansion(f, N,
                        G,
                        t=Matrix([[0,j/p],[j/p,k/p]]),
                        max_n=max_n,
                        boxes=G_box)
                    Tpf += f_jk / p^2
        elif b % p != 0:
            for i in range(p):
                for k in range(p):
                    f_ik = specialization_q_expansion(f, N,
                        G,
                        t=Matrix([[i/p,0],[0,k/p]]),
                        max_n=max_n,
                        boxes=G_box)
                    Tpf += f_ik / p^2
        elif c % p != 0:
            for i in range(p):
                for j in range(p):
                    f_ij = specialization_q_expansion(f, N,
                        G,
                        t=Matrix([[i/p,j/p],[j/p,0]]),
                        max_n=max_n,
                        boxes=G_box)
                    Tpf += f_ij / p^2
        else:
            print(f"Worst. case. ever. ({p})")
            for i in range(p):
                for j in range(p):
                    for k in range(p):
                        f_ijk = specialization_q_expansion(f, N,
                        G,
                        Matrix([[i/p,j/p],[j/p,k/p]]),
                        max_n=max_n,
                        boxes=G_box)

                        Tpf += f_ijk / p^3
        
        return Tpf
    except:
        print(f"Error: some Fourier coefficient is missing for T({p}). Try computing some more.")
        return specialization_q_expansion(f, N, s, max_n=max_n)


def Hecke_Tp_q_GritQ(Grits, f_G, s, p, N=277, max_n=20, warnings=None):
    a, b, c = s[0][0], s[0][1], s[1][1]*N
    G = p*s
    Grit_ps = [
        specialization_q_expansion(f, N,
        G,
        max_n=max_n, warnings=warnings)
        for f in Grits
    ]
    Tpf = p * f_G(Grit_ps)
    
    G = Matrix([[a/p, b], [b,p*c/N]])
    if a % p != 0:
        Grit_i = [
            specialization_q_expansion(f, N,
            G,
            max_n=max_n, warnings=warnings)
            for f in Grits
        ]
        Tpf += f_G(Grit_i)
    else:
        G_box = list(S_box(N, G, max_n, 0))
        for i in range(p):
            Grit_i = [
                specialization_q_expansion(f, N,
                G,
                t=Matrix([[i/p,0],[0,0]]),
                max_n=max_n,
                boxes=G_box)
                for f in Grits
            ]
            Tpf += f_G(Grit_i) / p

    ##########################

    for i in range(p):
        G = Matrix([[p*a, b+i*a], [b+i*a, (c/N + 2*i*b + i^2*a)/p]])
        G_box = list(S_box(N, G, max_n, 0))

        if (c + 2*i*b*N + i^2*a*N) % p != 0:
            Grit_i = [
                specialization_q_expansion(f, N,
                G,
                    max_n=max_n,
                    boxes=G_box)
                for f in Grits
            ]
            Tpf += f_G(Grit_i)
        else:
            for j in range(p):
                Grit_ij = [
                    specialization_q_expansion(f, N,
                    G,
                    t=Matrix([[0,0],[0,j/p]]),
                    max_n=max_n,
                    boxes=G_box)
                    for f in Grits
                ]
                Tpf += f_G(Grit_ij) / p

    ###########################


    G = s/p
    G_box = list(S_box(N, G, max_n, 0))
    if a % p != 0:
        for j in range(p):
            for k in range(p):
                Grit_jk = [
                    specialization_q_expansion(f, N,
                    G,
                    t=Matrix([[0,j/p],[j/p,k/p]]),
                    max_n=max_n,
                    boxes=G_box)
                    for f in Grits
                ]
                Tpf += f_G(Grit_jk) / p^2
    elif b % p != 0:
        for i in range(p):
            for k in range(p):
                Grit_ik = [
                    specialization_q_expansion(f, N,
                    G,
                    t=Matrix([[i/p,0],[0,k/p]]),
                    max_n=max_n,
                    boxes=G_box)
                    for f in Grits
                ]
                Tpf += f_G(Grit_ik) / p^2
    elif c % p != 0:
        for i in range(p):
            for j in range(p):
                Grit_ij = [
                    specialization_q_expansion(f, N,
                    G,
                    t=Matrix([[i/p,j/p],[j/p,0]]),
                    max_n=max_n,
                    boxes=G_box)
                    for f in Grits
                ]
                Tpf += f_G(Grit_ij) / p^2
    else:
        print(f"Worst. case. ever. ({p})")
        for i in range(p):
            for j in range(p):
                for k in range(p):
                    Grit_ijk = [
                        specialization_q_expansion(f, N,
                        G,
                        t=Matrix([[i/p,j/p],[j/p,k/p]]),
                        max_n=max_n,
                        boxes=G_box)
                        for f in Grits
                    ]
                    Tpf += f_G(Grit_ijk) / p^3
    
    return Tpf


def Hecke_Tp_box(s, p, N=277, max_n=3):
    a, b, c = s[0][0], s[0][1], s[1][1]*N
    
    box = list()

    G = p*s
    box += (list(S_box(N, G, max_n, 0)))

    G = Matrix([[a/p, b], [b,p*c/N]])
    box += (list(S_box(N, G, max_n, 0)))

    for i in range(p):
        G = Matrix([[p*a, b+i*a], [b+i*a, (c/N + 2*i*b + i^2*a)/p]])
        box += (list(S_box(N, G, max_n, 0)))

    G = s/p
    box += (list(S_box(N, G, max_n, 0)))

    reduced_box = []
    R = Matrix([[0,1],[1,0]])

    for detT, T in box:
        reduced_T, _, v = legendre_reduce(T, N=N)

        if v[0] != 0:
            u = v * mod(1/v[0], N)
            u = vector([Integer(u[0]),Integer(u[1])])
            U = Matrix([[u[0],0],[u[1],1]])
        else:
            u = v * mod(1/v[1], N)
            u = Integer(u)
            U = Matrix([[u[0],-1],[u[1],1]])

        reduced_box.append(
            [detT, R.T * U.T * reduced_T * U * R]
        )

    return reduced_box


def Hecke_Tp(f, p):
    Tpf = defaultdict(list)
    for determinant in f.keys():
        for T, _ in f[determinant]:
            a_T = 0
            
            if p^2 * determinant in f:
                for M, coefficient in f[p^2 * determinant]:
                    if M == p*T:
                        a_T += coefficient
                        break
            
            if determinant / p^2 in f:
                for M, coefficient in f[determinant / p^2]:
                    if M == (1/p) * T:
                        # p^{2k - 3} = p (k=2)
                        a_T += p * coefficient
                        break

            for j in range(p):
                u = Matrix([[1,0], [j, p]])
                uTu = u.T * T * u / p
                for M, coefficient in f[determinant]:
                    if M == uTu:
                        a_T += coefficient
                        break

            u = Matrix([[p,0], [0, 1]])
            uTu = u * T * u / p
            for M, coefficient in f[determinant]:
                if M == uTu:
                    a_T += coefficient
                    break
            
            if a_T != 0:
                Tpf[determinant].append([T, a_T])
        
    return Tpf
    

def tuple2matrix(T):
    return Matrix([
        [T[2], T[1]/2],
        [T[1]/2, T[0]]
    ])

def matrix2tuple(T):
    return (T[1][1], 2*T[0][1], T[0][0])
