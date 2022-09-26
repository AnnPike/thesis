import numpy as np
from mprod import  m_prod
from mprod import  generate_haar, generate_dct
import algo
import matplotlib.pyplot as plt
from scipy.stats import ortho_group



def bound(i, k, E0):
    return 2*E0*((np.sqrt(k)-1)/(np.sqrt(k)+1))**i


def nu_tensor_norm(E, A_tensor, funM, invM):
    return algo.tensor_frob_norm(algo.m_prod_three(E.transpose(1, 0, 2), A_tensor, E, funM, invM))


def generate_tall_matrix(m, p, eigenmin):
    A_o = ortho_group.rvs(dim=m)
    eigen = np.sqrt((np.random.rand(p)+1)*eigenmin)
    P = eigen.reshape(1, -1)*A_o[:, :p]
    Q = ortho_group.rvs(dim=p)
    A = np.matmul(P, Q.T)
    return A


def generate_tall_tensor_eigen(m, p, eigen1, eigen2, funM, invM):

    A1 = generate_tall_matrix(m, p, eigen1)
    A2 = generate_tall_matrix(m, p, eigen2)

    A_hat = np.concatenate([A1.reshape(m, p, 1), A2.reshape(m, p, 1)], 2)
    fun_dct, inv_dct = generate_dct(2)
    A_tall_tensor = invM(A_hat)

    X_true = np.random.randn(p, 1, 2)
    B = m_prod(A_tall_tensor, X_true, funM, invM)

    return A_tall_tensor, B, X_true


def generate_tensor_eigen(m, p, eigen1, eigen2, funM, invM):
    A1 = generate_tall_matrix(m, p, eigen1)
    A1 = np.matmul(A1.T, A1)
    A2 = generate_tall_matrix(m, p, eigen2)
    A2 = np.matmul(A2.T, A2)

    A_hat = np.concatenate([A1.reshape(p, p, 1), A2.reshape(p, p, 1)], 2)
    A_tensor = invM(A_hat)

    X_true = np.random.randn(p, 1, 2)
    B = m_prod(A_tensor, X_true, funM, invM)

    return A_tensor, B, X_true


m, p = 1000, 10
iters = 50
path_to_save = '/home/anna/uni/thesis/numerical_results/'


plt.figure(figsize=(15, 10))
for i in range(4):
    if i == 0:
        M_random = 'DCT'
        funM, invM = generate_dct(2)
    else:
        M_random = np.random.randint(1000)
        funM, invM = generate_haar(2, random_state=M_random)

    A_tensor, B, X_true = generate_tall_tensor_eigen(m, p, 1, 10**4, funM, invM)
    X, error = algo.LSQR_mprod(A_tensor, B, funM, invM, iters, X_true=X_true)
    bond_vector_f = np.vectorize(bound)
    bound_vector = bond_vector_f(np.arange(iters+1), 2*10**4, error[0])


    plt.subplot(2, 2, i+1)
    plt.plot(error, c='b', label='eigvals of Ahat^TA_hat for first slice are 1-2, second slice: 1*10^9-2*10^9')
    plt.plot(bound_vector, 'b--', label='bound')
    plt.xlabel('iterations')
    plt.yscale("log")

    A_tensor, B, X_true = generate_tall_tensor_eigen(m, p, 1, 1, funM, invM)
    X, error = algo.LSQR_mprod(A_tensor, B, funM, invM, iters, X_true=X_true)
    bound_vector = bond_vector_f(np.arange(iters+1), 2, error[0])
    plt.plot(error, c='y', label='all eigenvalues of Ahat^TA_hat are from 1 to 2')
    plt.plot(bound_vector, 'y--', label='bound')
    plt.xlabel('iterations')
    plt.yscale("log")
    plt.title(f'random state of M = {M_random}')
    plt.ylabel('error')
    plt.legend()
plt.suptitle('M prod LSQR')
plt.tight_layout()
plt.savefig(path_to_save+'eigenvalues_experiment_LSQR')
plt.show()

iters = 20
plt.figure(figsize=(15, 10))
for i in range(4):
    if i == 0:
        M_random = 'DCT'
        funM, invM = generate_dct(2)
    else:
        M_random = np.random.randint(1000)
        funM, invM = generate_haar(2, random_state=M_random)

    A_tensor, B, X_true = generate_tensor_eigen(m, p, 1, 10**4, funM, invM)
    X, error = algo.CG_M_tensor(A_tensor, B, funM, invM, iters, X_true=X_true)
    bond_vector_f = np.vectorize(bound)
    bound_vector = bond_vector_f(np.arange(iters+1), 2*10**4, error[0])


    plt.subplot(2, 2, i+1)
    plt.plot(error, c='b', label='eigvals of A^ for first slice are 1-2, second slice - 1*10^9-2*10^9')
    plt.plot(bound_vector, 'b--', label='bound')
    plt.xlabel('iterations')
    plt.yscale("log")

    A_tensor, B, X_true = generate_tensor_eigen(m, p, 1, 1, funM, invM)
    X, error = algo.CG_M_tensor(A_tensor, B, funM, invM, iters, X_true=X_true)
    bound_vector = bond_vector_f(np.arange(iters+1), 2, error[0])
    plt.plot(error, c='y', label='all eigenvalues of A^are from 1 to 2')
    plt.plot(bound_vector, 'y--', label='bound')
    plt.xlabel('iterations')
    plt.yscale("log")
    plt.title(f'random state of M = {M_random}')
    plt.ylabel('error')
    plt.legend()
plt.suptitle('M prod conjugate gradient')
plt.tight_layout()
plt.savefig(path_to_save+'eigenvalues_experiment_CG')
plt.show()

link = 'https://www.dropbox.com/scl/fo/vwn07zx7omd0zvqw0laxs/h?dl=0&rlkey=oyllju7bignxhr1liqdqh8b23'