from sage.modular.etaproducts import qexp_eta
from collections import defaultdict
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from itertools import product
import numpy as np
import scipy.sparse as sparse
load("siegel.sage")

def eta(prec=10):
    return qexp_eta(ZZ[['q']], prec)

def eta_power(d, k, prec_q=10):
    R.<z> = PowerSeriesRing(ZZ,sparse=True)
    S.<q> = PowerSeriesRing(R, sparse=True)

    if 2*k < len(d):
        eta_factor = (eta(prec=prec_q)**(-1))**(len(d) - 2*k)
    else:
        eta_factor = eta(prec=prec_q)**(2*k - len(d))
    
    eta_factor *= q**int(((k + len(d))/12))
    return S(eta_factor)

def zeta_half_product(d):
    R.<z> = LaurentSeriesRing(ZZ,sparse=True)
    A.<a> = PuiseuxSeriesRing(ZZ)

    half_product = 1
    for d_i in d:
        half_product *= a^(d_i/2) - a^(-d_i/2)

    whole_product = 0
    for e in range(-sum(d)/2, sum(d)/2 + 1):
        whole_product += half_product[e] * z^e

    return whole_product
    

def theta_function(d, prec_q=10, prec_z=10, return_d=False):
    R.<z> = PowerSeriesRing(ZZ,sparse=True)
    S.<q> = PowerSeriesRing(R,sparse=True)
        
    theta = 0
    n = 1
    zeta_sum = R(1)

    while binomial(n, 2) <= prec_q:
        if n > 1:
            zeta_sum += z^(d*(n-1)) + z^(d*(1-n))
        term = ((-1)^(n+1) * q^binomial(n,2)) * zeta_sum

        theta += term
        n += 1

    return theta


def multithread_theta_block(d, k=2, prec_q=10, prec_z=10, eta_factor=None, 
    theta_dict=dict(), tree_cache=dict()):
    R.<z> = LaurentSeriesRing(ZZ, sparse=True)
    S.<q> = PowerSeriesRing(R, sparse=True)

    n = defaultdict(int)
    for d_i in d:
        n[d_i] += 1
    short_d = set(d)

    with mp.Pool() as pool:
        factors = []
        for d_i in short_d:
            factors += [theta_dict[d_i]] * n[d_i]

        if eta_factor is None:
            if 2*k < len(d):
                eta_factor = (eta(prec=prec_q)^(-1))
                factors += [S(eta_factor)] * (len(d) - 2*k)
                # block *= eta_factor
            else:
                eta_factor = eta(prec=prec_q)
                factors += [S(eta_factor)] * (2*k - len(d))
                # block *= eta_factor
        else:
            factors += [eta_factor]

        while len(factors) > 1:
            splits = [factors[i:i+2] for i in range(0, len(factors), 2)]
            print(len(splits))
            # if len(splits[-1]) == 1:
                # last = splits.pop()[0]
                # splits[-1].append(last)
            
            factors = list(pool.imap_unordered(
                prod,
                splits,
            ))

    
    block = factors[0]
    block *= q^((k + len(d))/12)

    A.<a> = PuiseuxSeriesRing(QQ, sparse=True)

    zeta_half_product = 1
    for d_i in d:
        zeta_half_product *= a^(d_i/2) - a^(-d_i/2)

    zeta_whole_product = 0
    for e in range(-sum(d)/2, sum(d)/2 + 1):
        zeta_whole_product += zeta_half_product[e]*z^e
    block *= zeta_whole_product

    return block

def single_thread_theta_block(d, k=2, prec_q=10, prec_z=10, eta_factor=None, 
    theta_dict=dict(), tree_cache=dict()):
    R.<z> = PowerSeriesRing(ZZ, sparse=True)
    S.<q> = PowerSeriesRing(R, sparse=True)

    if eta_factor is None:
        # print("Computing eta product...")

        if 2*k < len(d):
            eta_factor = (eta(prec=prec_q)**(-1))**(len(d) - 2*k)
        else:
            eta_factor = eta(prec=prec_q)**(2*k - len(d))
    block = S(eta_factor)
    # block = S(1)

    for i, d_i in (enumerate(d)):
        if d_i not in theta_dict:
            theta_dict[d_i] = theta_function(d_i, prec_q=prec_q, prec_z=prec_z)
        block *= theta_dict[d_i]
        block = block.truncate_powerseries(prec_q)

    block *= q**int(((k + len(d))/12))

    A.<a> = PuiseuxSeriesRing(QQ)

    zeta_half_product = 1
    for d_i in d:
        zeta_half_product *= a^(d_i/2) - a^(-d_i/2)

    zeta_whole_product = 0
    for e in range(-sum(d)/2, sum(d)/2 + 1):
        zeta_whole_product += zeta_half_product[e]*z**e
    block *= zeta_whole_product

    return block


# def prod_coefficient(factors, n, r):
def prod_coefficient(factors, n):
    if len(factors) == 1:
        # return factors[0][n][r]
        return factors[0][n]
    else:
        f = factors[0]

        a = 0
        keys = f.dict().keys()
        for m, fm in f.dict().items():
            # print(m)
            if m > n:
                break
            elif fm == 0:
                continue
            else:
                # a += f[m][s] * prod_coefficient(factors[1:], n - m, r - s)
                a += fm * prod_coefficient(factors[1:], n - m)
        
        return a


def theta_block(*args, **kwargs):
    if "multithread" in kwargs:
        multithread = kwargs["multithread"]
        del kwargs["multithread"]
    else:
        multithread = False
    
    if multithread:
        return multithread_theta_block(*args, **kwargs)
    else:
        return single_thread_theta_block(*args, **kwargs)

def Jacobi_level_raising(n, r, m, TB, k=2):
    c = 0
    for d in divisors(gcd(gcd(n,r),m)):
        try:
            c += d^(k-1) * TB[m*n / d^2][r/d]
        except:
            print("Warning: coefficient not found")
            pass
    return c


def Gritsenko_lift(TB, T_list, N=277, k=2):
    f = defaultdict(list)
    for T in T_list:
        n, r, m = T[0][0], 2*T[0][1], T[1][1] / N
        f[det(2*T)].append([
            T, Jacobi_level_raising(n,r,m,TB, k=k)
        ])
    return f


def feasible_matrices(prec_q, prec_z, N):
    """Returns the maximal list of matrices that can appear
    in the Fourier-Jacobi expansion of a Gritsenko lift with the given precision."""
    for r in range(-prec_z, prec_z+1):
        for m in range(-prec_q, prec_q):
            if m == 0:
                continue
            t = abs(prec_q/m)
            for n in range(0, floor(t)+1):
                if 4*n*m*N - r^2 > 0:
                    yield tuple2matrix((N*m,r,n))


def polymul2d(a, b):
    """
    Performs FFT multiplication of a and b
    a, b must be 2D numpy arrays
    """

    shape = (a.shape[0] + b.shape[0], a.shape[1] + b.shape[1])

    a = np.fft.fft2(a, shape).astype("complex128")
    # print("First FFT done.")
    b = np.fft.fft2(b, shape).astype("complex128")
    # print("Second FFT done.")

    a *= b
    del b
    # print("Product done.")
    a = np.fft.ifft2(a)
    # print("IFFT done.")

    return a

def truncate_first_axis(a, n):
    a = a[:n+1]
    _, nonzero = a.nonzero()
    m = nonzero.max()
    a = a[:,:m+1]

    return a

def matrix_theta_function(di, prec_q=10):
    n = 1
    while binomial(n, 2) <= prec_q:
        shift = di * (n - 1)
        n += 1

    theta_matrix = np.zeros((binomial(n - 1,2) + 1, 2*shift + 1), dtype="int64")

    theta = 0
    n = 1
    zeta_sum = R(1)

    while binomial(n, 2) <= prec_q:
        q = binomial(n, 2)
        for z in range(- di * (n - 1), di * (n - 1) + 1, di):
            theta_matrix[q, shift + z] = (-1)^(n+1)

        n += 1
    return theta_matrix, shift


def matrix_eta(prec_q=10, power=1):
    eta_series = qexp_eta(ZZ[['q']], prec_q+1)
    eta_series = eta_series^power
    eta_matrix = np.zeros((prec_q + 1, 1))

    for i in range(prec_q):
        eta_matrix[i+1,0] = eta_series[i]

    return eta_matrix, 0


def matrix_baby_block(d):
    A.<a> = PuiseuxSeriesRing(ZZ)

    baby_block = 1
    for d_i in d:
        baby_block *= a^(d_i/2) - a^(-d_i/2)

    shift = sum(d) / 2
    matrix = np.zeros((1,2*shift + 1))
    for e in baby_block.exponents():
        matrix[0,shift + e] = baby_block[e]

    return matrix, shift


def matrix_theta_block(block, coefficients=set(), prec_q=10):
    acc, acc_shift = matrix_baby_block(block)
    # print("Baby block generated.")

    # add a zero to the left of eta
    # TODO: change this to the appropriate number of zeros
    # if k ever changes
    # eta, _ = matrix_eta(prec_q=prec_q, power=-6)

    # acc = polymul2d(acc, eta)
    # acc = acc.round()
    # acc = truncate_first_axis(acc, prec_q)
    # del eta

    for di in block:
        tdi, shift = matrix_theta_function(di, prec_q=prec_q)
        # print(f"Theta function {di} generated.")
        
        acc = polymul2d(acc, tdi)
        # print(f"Theta function {di} multiplied.")
        acc_shift += shift
        del tdi

        acc = acc.round()
        acc = truncate_first_axis(acc, prec_q)

    acc = acc.round().astype("int32")

    eta,_ = matrix_eta(prec_q=prec_q, power=-6)

    result = dict()

    for i in coefficients:
        res_i = acc[0,:] * eta[i,0]
        for j in range(i):
            res_i += acc[i - j] * eta[j,0]
        result[i] = res_i

        # print(res_i[acc_shift:])

    return result, acc_shift


def matrix_Jacobi_level_raising(n, r, m, TB, shift, k=2):
    c = 0
    for d in divisors(gcd(gcd(n,r),m)):
        try:
            c += d^(k-1) * Integer(TB[m*n / d^2][shift + (r/d)])
        except:
            # print(f"Warning: coefficient not found, {(m*n / d**2, r/d)}")
            pass
    return c


def matrix_Gritsenko_lift(TB, T_list, shift, N=277, k=2):
    f = defaultdict(list)
    for T in T_list:
        n, r, m = T[0][0], 2*T[0][1], T[1][1] / N
        f[det(2*T)].append([
            T, matrix_Jacobi_level_raising(n, r, m, TB, shift, k=k)
        ])
    return f
