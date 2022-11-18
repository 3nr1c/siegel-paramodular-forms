load("theta_blocks.sage")
import json
import sys


if __name__ == "__main__":

    N = 277
    max_n = 3
    max_n_hecke = 3

    if len(sys.argv) > 1:
        p = int(sys.argv[1])
    else:
        p = 2
    assert is_prime(p)

    found_m = False

    C = ceil(p^-1 * (1 + p) * (1 + p^2))
    M = 2*C + 1
    while not found_m:
        M = next_prime(M)
        F.<a> = CyclotomicField(p)
        found_m = (M % p == 1) and (len(F.primes_above(M)) == F.degree())

    print(f"Using m={M}")

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

    # T_list += list(feasible_matrices(prec_q, prec_z, N))
    # print(T_list)

    print("Computing eta inversion...")
    eta_factor = eta(prec=prec_q, m=M)^(-1)
    print("Computing eta power...")
    eta_factor = (eta_factor^2)^3
    eta_factor = eta_factor.truncate_powerseries(prec_q)

    G = []
    #TpG = []

    with mp.Pool(4) as pool:
        manager = mp.Manager()
        theta_dict = manager.dict()
        tree_cache = manager.dict()

        print("Computing theta blocks...")
        work = pool.imap(
            partial(theta_block, prec_q=prec_q, 
                prec_z=prec_z, eta_factor=eta_factor,
                theta_dict=theta_dict, tree_cache=tree_cache, m=M),
            block_ds_f_277
        )
        
        Gritsenko_lifts = []
        for i, TB in tqdm(enumerate(work), total=len(block_ds_f_277)):
            lift = Gritsenko_lift(TB, T_list)
            Gritsenko_lifts.append(lift)

            try:
                with open(f"tbs/Grit_TB_{i+1}.json") as json_file:
                    lift_dump = json.load(json_file)
            except:
                print(f"Creating file for Grit(TB{i})")
                lift_dump = {"Fourier_coefficients": dict()}
            for detG in lift:
                if str(detG) not in lift_dump["Fourier_coefficients"]:
                    lift_dump["Fourier_coefficients"][str(detG)] = dict()
                for T, aT in lift[detG]:
                    lift_dump["Fourier_coefficients"][str(detG)][str(matrix2tuple(T))] = str(aT)

            with open(f"tbs/Grit_TB_{i+1}.json", 'w') as f:
                json.dump(lift_dump, f, indent=4)

            special_lift = specialization_q_expansion(lift, N, s1, m=M, max_n=max_n, boxes=boxes)
            G.append(special_lift)
            
            #if p > 0:
            #    TpG.append(
            #        Hecke_Tp_q(lift, s1, p, N=N, max_n=max_n_hecke)
            #    )

            

    print(len(Gritsenko_lifts))
    print("All theta blocks computed.")
    # print("Computing lifts...")

    """
    for i, TB in tqdm(enumerate(theta_blocks_f_277)):
        lift = Gritsenko_lift(TB, T_list)

        try:
            with open(f"tbs/Grit_TB_{i+1}.json") as json_file:
                lift_dump = json.load(json_file)
        except:
            print(f"Creating file for Grit(TB{i})")
            lift_dump = {"Fourier_coefficients": dict()}
        for detG in lift:
            if str(detG) not in lift_dump["Fourier_coefficients"]:
                lift_dump["Fourier_coefficients"][str(detG)] = dict()
            for T, aT in lift[detG]:
                lift_dump["Fourier_coefficients"][str(detG)][str(matrix2tuple(T))] = str(aT)

        with open(f"tbs/Grit_TB_{i+1}.json", 'w') as f:
            json.dump(lift_dump, f, indent=4)

        special_lift = specialization_q_expansion(lift, 277, s1, max_n=max_n)
        G.append(special_lift)
    """

    # print(G)

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
        print(M)
        if L == 0:
            return L
        else:
            return L/M

    L.<q> = PuiseuxSeriesRing(ZZ)
    ff = f277(G)
    f = L(0)
    for e in ff.exponents():
        coeff = ZZ(ff[e].lift())
        while coeff > C:
            coeff -= M
        while coeff < -C:
            coeff += M
        f += coeff * q^e

    print(f.add_bigoh(max_n+1))


    if p > 0:
        Tpff = (Hecke_Tp_q_GritQ(Gritsenko_lifts, f277, s1, p, m=M, max_n=max_n_hecke))

        Tpf = L(0)
        for e in Tpff.exponents():
            coeff = ZZ(Tpff[e].lift())
            while coeff > C:
                coeff -= M
            while coeff < -C:
                coeff += M
            Tpf += coeff * q^e
        print(Tpf.add_bigoh(max_n+1))