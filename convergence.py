import numpy as np
from mprod import  m_prod
from mprod import  generate_haar, generate_dct
import algo
import matplotlib.pyplot as plt


def generate_equation(m, p, n, ran_int, psd):
    np.random.seed(ran_int)
    A_orig = np.random.rand(m, p, n)
    funM, invM = generate_haar(n, random_state=ran_int)
    if psd:
        A_tensor = m_prod(A_orig.transpose((1, 0, 2)), A_orig, funM, invM)+10*np.array(n*[np.identity(p)]).transpose((1,2,0))
    else:
        A_tensor = A_orig.copy()
    X_true = np.random.rand(p, 1, n)

    B = m_prod(A_tensor, X_true, funM, invM)
    return A_tensor, B, funM, invM, X_true


path_to_save = '/home/anna/uni/thesis/numerical_results/'
def plot_convergence(algorithm, m, p, n, psd, path_to_save=path_to_save):
    plt.figure(figsize=(10, 5))
    np.random.seed(0)
    for i in range(4):
        ran_int = np.random.randint(1000)
        A_tensor, B, funM, invM, X_true = generate_equation(m, p, n, ran_int, psd)
        X, error = algorithm(A_tensor, B, funM, invM, 25, X_true=X_true)
        plt.plot(error, label=f'random state M = {ran_int}')
        plt.yscale("log")
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('error')
    algo_name = str(algorithm)[10:-18]
    plt.suptitle(f'{algo_name} height={m}, width={p}, depth={n}')
    plt.tight_layout()
    plt.savefig(f'{path_to_save}{algo_name}')
    plt.show()


m, p, n = 500, 50, 2
plot_convergence(algo.CG_M_tensor, m, p, n, True)
plot_convergence(algo.LSQR_mprod, m, p, n, False)

