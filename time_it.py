import numpy as np
from mprod import  m_prod
from mprod import  generate_haar, generate_dct
import algo
from scipy.stats import ortho_group
import helper_plot
import matplotlib.pyplot as plt
import time
import pickle
import gc



path_to_save = 'numerical_results/DCTvsS/'
start_blendenpik = time.time()


gama = 4
for n in [3, 10, 15]:

    funM, invM = generate_haar(n, random_state=21)

    if n == 3:
        start = 26000
        time_S_list = pickle.load(open(f'{path_to_save}S_list_n{n}', 'rb'))
        time_b_list = pickle.load(open(f'{path_to_save}b_list_n{n}', 'rb'))
    else:
        start = 500
        time_S_list = []
        time_b_list = []

    for i, m in enumerate(range(start, 30001, 500)):
        np.random.seed(0)
        p = m//20
        s = gama*p
        tensor_A = np.random.randn(m, p, n)
        # B = np.random.randn(m, 1, n)

        start_time_S = time.time()
        P, R = algo.sampling_QR(tensor_A, funM, invM, s)
        # X, error, list_of_X = algo.LSQR_mprod_tuples_precond(tensor_A, B, R, funM, invM, 10, X_true=None)
        time_S = time.time()-start_time_S
        time_S_list.append(time_S)

        start_time_b = time.time()
        P, R = algo.blendenpick(tensor_A, funM, invM, s=s)
        # X, error, list_of_X = algo.LSQR_mprod_tuples_precond(tensor_A, B, R, funM, invM, 10, X_true=None)
        time_b = time.time() - start_time_b
        time_b_list.append(time_b)

        pickle.dump(time_S_list, open(f'{path_to_save}S_list_n{n}', 'wb'))
        pickle.dump(time_b_list, open(f'{path_to_save}b_list_n{n}', 'wb'))
        print(f'finished n={n}, m={m}')

        if i != 0:
            plt.figure(figsize=(10, 5))
            plt.plot(time_S_list, 'g', label='S ~ N(1,0)')
            plt.plot(time_b_list, 'b', label='dct')
            plt.legend()
            plt.title(f'Time, p=m//20, n={n}, s=4*p')
            plt.ylabel('seconds')
            plt.xlabel('m')
            plt.xticks(range(0, len(time_S_list), 10), range(500, len(time_S_list)*500+1, 500*10))
            plt.savefig(f'{path_to_save}timeit_n{n}')

            plt.close()
            gc.collect()
            time.sleep(10)


plt.figure(figsize=(10, 5))
colors = {3: 'purple', 10: 'b', 15: 'g'}
for n in [3, 10, 15]:
    lS = pickle.load(open(f'{path_to_save}S_list_n{n}', 'rb'))
    ld = pickle.load(open(f'{path_to_save}b_list_n{n}', 'rb'))
    plt.plot(lS, label=f'S ~ N(1,0), n={n}', ls='--', c=colors[n])
    plt.plot(ld, label=f'dct, n={n}', c=colors[n])

plt.title(f'Time, p=m//20, s=4*p')
plt.ylabel('seconds')
plt.xlabel('m')
plt.legend()
plt.savefig('timeit_over_n')
plt.show()

