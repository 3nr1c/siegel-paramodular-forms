import multiprocessing as mp
from functools import partial
load("siegel.sage")

if __name__ == "__main__":

    R.<x> = QQ[]
    curves = {
        277: [-x^2 - x, x^3 + x^2 + x + 1],
        349: [-x^3 - x^2, x^3 + x^2 + x + 1],
        353: [x^2, x^3+x+1],
        #389: [x^5 - 2*x^4 - 8*x^3 + 16*x + 7, x^3 + x]
    }

    max_n = 3

    for N in [277]: # 277, 349, 353, 389, 461, 523]:
        print(f"N = {N}")
        f = read_coefficients(f"lmfdb_data/Kp.2_PY2_{N}.json")

        for detG in sorted(f.keys()):
            for T0, aT in f[detG]:
                if aT != 0:
                    break
            if aT != 0:
                break

        s1 = 2*T0.adjugate()
        print(f"det(2T0) = {detG}, a(T0;f) = {aT}")

        qexp = specialization_q_expansion(f, N, s1, max_n=max_n, warnings=True)
        print(qexp)
        print()

        F,G = curves[N]
        C = HyperellipticCurve(F,G)

        with mp.Pool(3) as pool:
            P = list(primes(2, 12))
            work = pool.imap(partial(Hecke_Tp_q, f, s1, N=N, max_n=max_n, warnings=True), P)

            for p, Tp_qexp in zip(P, work):
                print(f"p = {p}")
                print(Tp_qexp)
                print(f"a_p(f) = {Tp_qexp[detG] / qexp[detG]}")
                print(f"a_p(C) = {-C.change_ring(GF(p)).frobenius_polynomial()[3]}")
                print()

        print("\n------------------\n")