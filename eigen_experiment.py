import numpy as np
np.random.seed(0)
from mprod import  m_prod
from mprod import  generate_haar, generate_dct
import algo
import matplotlib.pyplot as plt
from scipy.stats import ortho_group



def bound(i, k, E0):
    return 2*E0*((np.sqrt(k)-1)/(np.sqrt(k)+1))**i


bond_vector_f = np.vectorize(bound)


def nu_tensor_norm(E, A_tensor, funM, invM):
    return algo.tensor_frob_norm(algo.m_prod_three(E.transpose(1, 0, 2), A_tensor, E, funM, invM))


def generate_tall_matrix(m, p, eigenmin):
    A_o = ortho_group.rvs(dim=m)
    eigen = np.sqrt((np.random.rand(p)+1)*eigenmin)
    P = eigen.reshape(1, -1)*A_o[:, :p]
    Q = ortho_group.rvs(dim=p)
    A = np.matmul(P, Q.T)
    return A


def generate_tall_A_hat(m, p, eigen1, eigen2):

    A1 = generate_tall_matrix(m, p, eigen1)
    A2 = generate_tall_matrix(m, p, eigen2)

    A_hat = np.concatenate([A1.reshape(m, p, 1), A2.reshape(m, p, 1)], 2)
    return A_hat


def generate_tall_A(A_hat, transform, randomM, X_true):
    if randomM!='DCT':
        funM, invM = generate_haar(2, random_state=randomM)
    else:
        funM, invM = generate_dct(2)
    if transform == 'DCT':
        fun_dct, inv_dct = generate_dct(2)
        A_tall_tensor = inv_dct(A_hat)
    else:
        A_tall_tensor = invM(A_hat)
    B = m_prod(A_tall_tensor, X_true, funM, invM)
    return A_tall_tensor, B


# def generate_tall_tensor_eigen(m, p, eigen1, eigen2, funM, invM, add_dct):
#     np.random.seed(0)
#     A1 = generate_tall_matrix(m, p, eigen1)
#     A2 = generate_tall_matrix(m, p, eigen2)
#
#     A_hat = np.concatenate([A1.reshape(m, p, 1), A2.reshape(m, p, 1)], 2)
#     fun_dct, inv_dct = generate_dct(2)
#     A_tall_tensor = invM(A_hat)
#     if add_dct:
#         A_tall_tensor = fun_dct(A_tall_tensor)
#     B = m_prod(A_tall_tensor, X_true, funM, invM)
#     return A_tall_tensor, B


# def generate_tensor_eigen(m, p, eigen1, eigen2, funM, invM, add_dct):
#     np.random.seed(0)
#     A1 = generate_tall_matrix(m, p, eigen1)
#     A1 = np.matmul(A1.T, A1)
#     A2 = generate_tall_matrix(m, p, eigen2)
#     A2 = np.matmul(A2.T, A2)
#
#     A_hat = np.concatenate([A1.reshape(p, p, 1), A2.reshape(p, p, 1)], 2)
#     fun_dct, inv_dct = generate_dct(2)
#     A_tensor = invM(A_hat)
#     if add_dct:
#         A_tensor = fun_dct(A_tensor)
#     B = m_prod(A_tensor, X_true, funM, invM)
#     return A_tensor, B



def generate_square_A(A_hat, transform, randomM, X_true):
    if randomM != 'DCT':
        funM, invM = generate_haar(2, random_state=randomM)
    else:
        funM, invM = generate_dct(2)
    if transform == 'DCT':
        fun_dct, inv_dct = generate_dct(2)
        A_tall_tensor = inv_dct(A_hat)
    else:
        A_tall_tensor = invM(A_hat)
    A_tensor = m_prod(A_tall_tensor.transpose(1, 0, 2), A_tall_tensor, funM, invM)
    B = m_prod(A_tensor, X_true, funM, invM)
    return A_tensor, B


path_to_save = '/home/anna/uni/thesis/numerical_results/'
m, p = 500, 50
X_true = np.random.randn(p, 1, 2)
M_list = ['DCT', 21, 127, 333]


fig = plt.figure(figsize=(15, 10))
#LSQR parameters
A_tall_hat_bad = generate_tall_A_hat(m, p, eigen1=1, eigen2=10**9)
A_tall_hat_good = generate_tall_A_hat(m, p, eigen1=1, eigen2=1)
iters = 40

for i in range(4):
    M_random = M_list[i]
    if M_random == 'DCT':
        funM, invM = generate_dct(2)
    else:
        funM, invM = generate_haar(2, random_state=M_random)

    plt.subplot(2, 2, i+1)

    A_tensor, B = generate_tall_A(A_tall_hat_bad, 'original M', M_random, X_true)
    X, error = algo.LSQR_mprod(A_tensor, B, funM, invM, iters, X_true=X_true)
    bound_vector = bond_vector_f(np.arange(iters+1), 2*10**4, error[0])

    l1 = plt.plot(error, c='b', label='eigvals of Ahat^TA_hat for first slice are 1-2, second slice: 1*10^9-2*10^9')
    # plt.plot(bound_vector, 'b--', label='bound')


    A_tensor, B = generate_tall_A(A_tall_hat_good, 'original M', M_random, X_true)
    X, error = algo.LSQR_mprod(A_tensor, B, funM, invM, iters, X_true=X_true)
    bound_vector = bond_vector_f(np.arange(iters+1), 2, error[0])

    l2 = plt.plot(error, c='y', label='all eigenvalues of Ahat^TA_hat are from 1 to 2')
    # plt.plot(bound_vector, 'y--', label='bound')

    A_tensor, B = generate_tall_A(A_tall_hat_bad, 'DCT', M_random, X_true)
    X, error = algo.LSQR_mprod(A_tensor, B, funM, invM, iters, X_true=X_true)
    bound_vector = bond_vector_f(np.arange(iters + 1), 2 * 10 ** 4, error[0])

    l3 = plt.plot(error, c='purple', label='eigvals of Ahat^TA_hat for first slice are 1-2, second slice: 1*10^9-2*10^9 transformed DCT')

    A_tensor, B = generate_tall_A(A_tall_hat_good, 'DCT', M_random, X_true)
    X, error = algo.LSQR_mprod(A_tensor, B, funM, invM, iters, X_true=X_true)
    bound_vector = bond_vector_f(np.arange(iters + 1), 2 * 10 ** 4, error[0])

    l4 = plt.plot(error, c='orange', label='all eigenvalues of Ahat^TA_hat are from 1 to 2 trabsformed DCT')



    plt.xlabel('iterations')
    plt.yscale("log")
    plt.title(f'random state of M = {M_random}')
    plt.ylabel('error')
    # plt.legend(loc='upper right')

line_labels = ['eigvals of Ahat^TA_hat for first slice are 1-2, second slice: 1*10^9-2*10^9',
               'all eigenvalues of Ahat^TA_hat are from 1 to 2',
               'eigvals of Ahat^TA_hat for first slice are 1-2, second slice: 1*10^9-2*10^9 transformed DCT',
               'all eigenvalues of Ahat^TA_hat are from 1 to 2 trabsformed DCT']
fig.legend([l1, l2, l3, l4], labels=line_labels, bbox_to_anchor=(0.8, 0.9))
plt.suptitle('M prod LSQR')
plt.tight_layout()
plt.savefig(path_to_save+'eigenvalues_experiment_LSQR_trans_dct')
plt.show()


iters = 20

fig = plt.figure(figsize=(15, 10))
for i in range(4):
    M_random = M_list[i]
    if M_random == 'DCT':
        funM, invM = generate_dct(2)
    else:
        funM, invM = generate_haar(2, random_state=M_random)

    plt.subplot(2, 2, i + 1)

    A_tensor, B = generate_square_A(A_tall_hat_bad, 'original M', M_random, X_true)
    X, error = algo.CG_M_tensor(A_tensor, B, funM, invM, iters, X_true=X_true)
    bound_vector = bond_vector_f(np.arange(iters+1), 2*10**4, error[0])

    l1 = plt.plot(error, c='b', label='eigvals of A^ for first slice are 1-2, second slice - 1*10^9-2*10^9')
    # plt.plot(bound_vector, 'b--', label='bound')

    A_tensor, B = generate_square_A(A_tall_hat_good, 'original M', M_random, X_true)
    X, error = algo.CG_M_tensor(A_tensor, B, funM, invM, iters, X_true=X_true)
    # bound_vector = bond_vector_f(np.arange(iters+1), 2, error[0])

    l2 = plt.plot(error, c='y', label='all eigenvalues of A^are from 1 to 2')
    # plt.plot(bound_vector, 'y--', label='bound')

    A_tensor, B = generate_square_A(A_tall_hat_bad, 'DCT', M_random, X_true)
    X, error = algo.CG_M_tensor(A_tensor, B, funM, invM, iters, X_true=X_true)
    bound_vector = bond_vector_f(np.arange(iters+1), 2*10**4, error[0])


    l3 = plt.plot(error, c='purple', label='eigvals of A^ for first slice are 1-2, second slice - 1*10^9-2*10^9 transformed DCT')

    A_tensor, B = generate_square_A(A_tall_hat_good, 'DCT', M_random, X_true)
    X, error = algo.CG_M_tensor(A_tensor, B, funM, invM, iters, X_true=X_true)
    bound_vector = bond_vector_f(np.arange(iters+1), 2, error[0])

    l4 = plt.plot(error, c='orange', label='all eigenvalues of A^are from 1 to 2 transformed DCT')

    plt.xlabel('iterations')
    plt.yscale("log")
    plt.title(f'random state of M = {M_random}')
    plt.ylabel('error')

fig.legend([l1, l2, l3, l4], labels=line_labels, bbox_to_anchor=(0.8, 0.9))
plt.suptitle('M prod conjugate gradient')
plt.tight_layout()
plt.savefig(path_to_save+'eigenvalues_experiment_CG_add_trans_dct')
plt.show()

link = 'https://www.dropbox.com/scl/fo/vwn07zx7omd0zvqw0laxs/h?dl=0&rlkey=oyllju7bignxhr1liqdqh8b23'