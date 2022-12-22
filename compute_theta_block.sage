load("theta_blocks.sage")
import json
import sys

if __name__ == "__main__":

    N = 277
    max_c = 130000
    max_det = 250

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

    print(f"Computing with det(2T) up to {max_det}.")

    print(f"Precomputing Grit boxes...")
    T_list = list()

    prec_q = 1

    for T in abc_box(max_c, max_det=max_det, N=N):
        n, r, m = T[0][0], 2*T[0][1], T[1][1] / N
        T_list.append(T)
        prec_q = max(prec_q, m*n)
    print(prec_q)

    print("Computing eta inversion...")
    eta_factor = eta(prec=prec_q)^(-1)
    print("Computing eta power...")
    eta_factor = (eta_factor^2)^3
    eta_factor = eta_factor.truncate_powerseries(prec_q)


    print("Setting things up...")
    manager = mp.Manager()
    theta_dict = manager.dict()
    tree_cache = manager.dict()

    for i in range(len(block_ds_f_277)):
        print(f"Computing block #{i}.")
        TB = ThetaBlock(block_ds_f_277[i], prec_q=prec_q, eta_factor=eta_factor, m=100)

        print("Block has been computed.")
        print("Computing Gritsenko lift...")

        lift = Gritsenko_lift(TB, T_list)

        print("Lift has been computed.")
        print("Saving...")

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

        print("Block was saved.")
