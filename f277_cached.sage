load("theta_blocks.sage")
import json
import sys

if __name__ == "__main__":


    N = 277
    max_n = 3
    max_n_hecke = 9

    if len(sys.argv) > 1:
        p = int(sys.argv[1])
    else:
        p = 2
    assert is_prime(p)


    block_ds_f_277 = [
        [2,4,4,4,5,6,8,9,10,14],
        [2,3,4,5,5,7,7,9,10,14],
        [2,3,4,4,5,7,8,9,11,13],
        [2,3,3,5,6,6,8,9,11,13],
        [2,3,3,5,5,8,8,8,11,13],
        [2,3,3,5,5,7,8,10,10,13],
        [2,3,3,4,5,6,7,9,10,15],
        [2,2,4,5,6,7,7,9,11,13],
        [2,2,4,4,6,7,8,10,11,12],
        [2,2,3,5,6,7,9,9,11,12]
    ]

    for d in block_ds_f_277:
        if (sum([di^2 for di in d]) / 2) != N:
            print(d)
            exit("Error: some theta block will yield an invalid level")

    theta_blocks_f_277 = []

    s1 = 2*Matrix(tuple2matrix((13573, 233, 1))).adjugate()
    boxes = list(S_box(N, s1, max_n, 0))
    T_list = [T for _, T in boxes]
    if p>0:
        print(f"Computing T({p})")
        T_list += [T for _, T in Hecke_Tp_box(s1, p, max_n=max_n_hecke)]

    for T in T_list:
        print(matrix2tuple(T))

    prec_q = 1
    prec_z = ceil(2*sqrt(N * 10))

    coefficients = set()
    for T in T_list:
        n, r, m = T[0][0], 2*T[0][1], T[1][1] / N
        prec_q = max(prec_q, m*n)
        prec_z = max(prec_z, abs(r))
        for d in divisors(gcd(m,n)):
            coefficients.add(m*n / d^2)
    print(prec_q, prec_z)
    print(coefficients)


    G = []
    Gritsenko_lifts = []
    for i in range(1, 11):
        lift = read_coefficients(f"tbs/Grit_TB_{i}.json")
        Gritsenko_lifts.append(lift)

        special_lift = specialization_q_expansion(lift, N, s1, max_n=max_n, boxes=boxes)
        G.append(special_lift)

    
    def f277(G):
        L = (-14 * G[0]^2-20 * G[7] * G[1]+11 * G[8] * G[1]+6 * G[1]^2-30 * G[6] * G[9]+15 * G[8] * G[9] 
        + 15 * G[9] * G[0]-30 * G[9] * G[1]-30 * G[9] * G[2]+5 * G[3] * G[4]+6 * G[3] * G[5]+17 * G[3] * G[6] 
        -3 * G[3] * G[7]-5 * G[3] * G[8]-5 * G[4] * G[5]+20 * G[4] * G[6]-5 * G[4] * G[7]-10 * G[4] * G[8]-3 * G[5]^2 
        +13 * G[5] * G[6]+3 * G[5] * G[7]-10 * G[5] * G[8]-22 * G[6]^2+G[6] * G[7]+15 * G[6] * G[8]+6 * G[7]^2 
        -4 * G[7] * G[8]-2 * G[8]^2+20 * G[0] * G[1]-28 * G[2] * G[1]+23 * G[3] * G[1]+7 * G[5] * G[1] 
        -31 * G[6] * G[1]+15 * G[4]* G[1]+45 * G[0] * G[1]-10 * G[0] * G[4]-2 * G[0] * G[3]-13 * G[0] * G[5] 
        -7 * G[0] * G[7]+39 * G[0] * G[6]-16 * G[0] * G[8]-34 * G[2]^2+8* G[2] * G[3]+20 * G[2] * G[4] 
        +22 * G[2] * G[5]+10 * G[2] * G[7]+21 * G[2] * G[8]-56 * G[2] * G[6]-3 * G[3]^2)
        M = (-G[3]+G[5]+2 * G[6]+G[7]-G[8]+2 * G[2]-3 * G[1]-G[0])
        if L == 0:
            return L
        else:
            return L/M

    L.<q> = ZZ[]
    ff = f277(G)

    print(ff)


    if p > 0:
        Tpff = (Hecke_Tp_q_GritQ(Gritsenko_lifts, f277, s1, p, max_n=max_n_hecke))

        print(Tpff)