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
    H = ortho_group.rvs(dim=m)
    eigen = np.sqrt((np.random.rand(p)+1)*eigenmin)
    # P = eigen.reshape(1, -1)*A_o[:, :p]
    LAM = np.zeros((p, p))
    np.fill_diagonal(LAM, eigen)
    P = np.matmul(H[:, :p], LAM)
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
#
#
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

def res_norm(A, X, B, funM, invM):
    return algo.tensor_frob_norm(m_prod(A, X, funM, invM)-B)


def res_norm_steps(A, list_of_X, B, funM, invM):
    list_of_res = []
    for i in range(len(list_of_X)):
        current_X = list_of_X[i]
        list_of_res.append(res_norm(A, current_X, B, funM, invM))
    return list_of_res


def norm_res_norm_steps(A, list_of_X, B, funM, invM):
    list_of_res = []
    for i in range(len(list_of_X)):
        current_X = list_of_X[i]
        residual = m_prod(A, current_X, funM, invM)-B
        residual_T = algo.tensor_frob_norm(m_prod(A.transpose(1,0,2), residual, funM, invM))
        norm_res = residual_T/(algo.tensor_frob_norm(A)*algo.tensor_frob_norm(residual))
        list_of_res.append(norm_res)
    return list_of_res



def get_eigen_sym(A, funM):
    A_hat = funM(A)
    lambda1 = np.linalg.eigvals(A_hat[:, :, 0])
    a1_min = np.round(lambda1.min(), 5)
    a1_max = np.round(lambda1.max(), 5)
    lambda2 = np.linalg.eigvals(A_hat[:, :, 1])
    a2_min = np.round(lambda2.min(), 5)
    a2_max = np.round(lambda2.max(), 5)
    return a1_min, a1_max, a2_min, a2_max

def get_eigen_tall(A, funM):
    A_hat = funM(A)
    A_hat_sym = np.einsum('mpi,pli->mli', A_hat.transpose(1, 0, 2), A_hat)
    lambda1 = np.linalg.eigvals(A_hat_sym[:, :, 0])
    a1_min = np.round(lambda1.min(), 5)
    a1_max = np.round(lambda1.max(), 5)
    lambda2 = np.linalg.eigvals(A_hat_sym[:, :, 1])
    a2_min = np.round(lambda2.min(), 5)
    a2_max = np.round(lambda2.max(), 5)

    return a1_min, a1_max, a2_min, a2_max

path_to_save = '/home/anna/uni/thesis/numerical_results/'
m, p = 500, 50
X_true = np.random.randn(p, 1, 2)
M_list = ['DCT', 21, 127, 333]
degree = 9
A_tall_hat_bad = generate_tall_A_hat(m, p, eigen1=1, eigen2=10**degree)
A_tall_hat_good = generate_tall_A_hat(m, p, eigen1=1, eigen2=1)

fig = plt.figure(figsize=(15, 10))
#LSQR parameters
iters = 40



for i in range(4):
    M_random = M_list[i]
    if M_random == 'DCT':
        funM, invM = generate_dct(2)
    else:
        funM, invM = generate_haar(2, random_state=M_random)

    plt.subplot(2, 2, i+1)

    A_tensor, B = generate_tall_A(A_tall_hat_bad, 'original M', M_random, X_true)
    X, error, list_of_X = algo.LSQR_mprod_tuples(A_tensor, B, funM, invM, iters, X_true=X_true)
    bound_vector = bond_vector_f(np.arange(iters+1), 2*10**degree, error[0])

    a1_min, a1_max, a2_min, a2_max = get_eigen_tall(A_tensor, funM)
    # l1 = plt.plot(error, c='b',
    #               label=f'$min(\lambda_1)$={a1_min:.0e}, $max(\lambda_1)$={a1_max:.0e}\n$min(\lambda_2)$={a2_min:.0e}, $max(\lambda_2)$={a2_max:.0e}')
    # plt.plot(bound_vector, 'b--', label='bound')
    res = norm_res_norm_steps(A_tensor, list_of_X, B, funM, invM)
    l1 = plt.plot(res, c='b',
                  label=f'$min(\lambda_1)$={a1_min:.0e}, $max(\lambda_1)$={a1_max:.0e}\n$min(\lambda_2)$={a2_min:.0e}, $max(\lambda_2)$={a2_max:.0e}')


    A_tensor, B = generate_tall_A(A_tall_hat_good, 'original M', M_random, X_true)
    X, error, list_of_X = algo.LSQR_mprod_tuples(A_tensor, B, funM, invM, iters, X_true=X_true)
    bound_vector = bond_vector_f(np.arange(iters+1), 2, error[0])
    a1_min, a1_max, a2_min, a2_max = get_eigen_tall(A_tensor, funM)
    # l2 = plt.plot(error, c='y', label=f'$min(\lambda_1)$={a1_min:.0e}, $max(\lambda_1)$={a1_max:.0e}\n$min(\lambda_2)$={a2_min:.0e}, $max(\lambda_2)$={a2_max:.0e}')
    # plt.plot(bound_vector, 'y--', label='bound')
    res = norm_res_norm_steps(A_tensor, list_of_X, B, funM, invM)
    l2 = plt.plot(res, c='y',
                  label=f'$min(\lambda_1)$={a1_min:.0e}, $max(\lambda_1)$={a1_max:.0e}\n$min(\lambda_2)$={a2_min:.0e}, $max(\lambda_2)$={a2_max:.0e}')
    s = 300

    A_tensor, B = generate_tall_A(A_tall_hat_bad, 'original M', M_random, X_true)
    P, R = algo.sampling_QR(A_tensor, funM, invM, s=s)
    A_tensor_precond = m_prod(A_tensor, R, funM, invM)
    # X_true_new = m_prod(P, X_true, funM, invM)
    X, error, list_of_X = algo.LSQR_mprod_tuples_precond(A_tensor, B, R, funM, invM, iters, X_true=X_true)
    bound_vector = bond_vector_f(np.arange(iters + 1), 2 * 10 ** 4, error[0])
    a1_min, a1_max, a2_min, a2_max = get_eigen_tall(A_tensor_precond, funM)
    # l3 = plt.plot(error, c='purple', label=f'BLENDENPIK\n$min(\lambda_1)$={a1_min:.0e}, $max(\lambda_1)$={a1_max:.0e}\n$min(\lambda_2)$={a2_min:.0e}, $max(\lambda_2)$={a2_max:.0e}')
    res = norm_res_norm_steps(A_tensor, list_of_X, B, funM, invM)
    l3 = plt.plot(res, c='purple',
                  label=f'$min(\lambda_1)$={a1_min:.0e}, $max(\lambda_1)$={a1_max:.0e}\n$min(\lambda_2)$={a2_min:.0e}, $max(\lambda_2)$={a2_max:.0e}\nBLENDENPIK')
    A_tensor, B = generate_tall_A(A_tall_hat_good, 'original M', M_random, X_true)
    P, R = algo.sampling_QR(A_tensor, funM, invM, s=s)
    A_tensor_precond = m_prod(A_tensor, R, funM, invM)
    # X_true_new = m_prod(P, X_true, funM, invM)
    X, error, list_of_X = algo.LSQR_mprod_tuples_precond(A_tensor, B, R, funM, invM, iters, X_true=X_true)
    bound_vector = bond_vector_f(np.arange(iters + 1), 2 * 10 ** 4, error[0])
    a1_min, a1_max, a2_min, a2_max = get_eigen_tall(A_tensor_precond, funM)
    # l4 = plt.plot(error, c='orange', label=f'BLENDENPIK\n$min(\lambda_1)$={a1_min:.0e}, $max(\lambda_1)$={a1_max:.0e}\n$min(\lambda_2)$={a2_min:.0e}, $max(\lambda_2)$={a2_max:.0e}')
    res = norm_res_norm_steps(A_tensor, list_of_X, B, funM, invM)
    l4 = plt.plot(res, c='orange',
                  label=f'$min(\lambda_1)$={a1_min:.0e}, $max(\lambda_1)$={a1_max:.0e}\n$min(\lambda_2)$={a2_min:.0e}, $max(\lambda_2)$={a2_max:.0e}\nBLENDENPIK')

    plt.legend(loc='upper right')
    plt.xlabel('iterations')
    plt.yscale("log")
    plt.title(f'random state of M = {M_random}')
    plt.ylabel('normalized A^T*residual')
    # plt.legend(loc='upper right')
#
line_labels = [f'eigvals of Ahat^TA_hat for first slice are 1-2, second slice: 1*10^{degree}-2*10^{degree}',
               'all eigenvalues of Ahat^TA_hat are from 1 to 2',
               f'eigvals of Ahat^TA_hat for first slice are 1-2, second slice: 1*10^{degree}-2*10^{degree}',
               'all eigenvalues of Ahat^TA_hat are from 1 to 2']
fig.legend([l1, l2, l3, l4], labels=line_labels, bbox_to_anchor=(0.8, 0.7))
plt.suptitle(f'M prod LSQR, A shape is {m}, {p}, 2, s={s}')
plt.tight_layout()
plt.savefig(path_to_save+f'eigenvalues_experiment_LSQR_blendenpik_{s}_res_norm')
plt.show()


iters = 30

# fig = plt.figure(figsize=(15, 10))
# for i in range(4):
#     M_random = M_list[i]
#     if M_random == 'DCT':
#         funM, invM = generate_dct(2)
#     else:
#         funM, invM = generate_haar(2, random_state=M_random)
#
#     plt.subplot(2, 2, i + 1)
#
#     A_tensor, B = generate_square_A(A_tall_hat_bad, 'original M', M_random, X_true)
#     X, error = algo.CG_M_tensor(A_tensor, B, funM, invM, iters, X_true=X_true)
#     bound_vector = bond_vector_f(np.arange(iters+1), 2*10**4, error[0])
#
#     a1_min, a1_max, a2_min, a2_max = get_eigen_sym(A_tensor, funM)
#     l1 = plt.plot(error, c='b', label=f'$min(\lambda_1)$={a1_min:.0e}, $max(\lambda_1)$={a1_max:.0e}\n$min(\lambda_2)$={a2_min:.0e}, $max(\lambda_2)$={a2_max:.0e}')
#     # plt.plot(bound_vector, 'b--', label='bound')
#
#     A_tensor, B = generate_square_A(A_tall_hat_good, 'original M', M_random, X_true)
#     X, error = algo.CG_M_tensor(A_tensor, B, funM, invM, iters, X_true=X_true)
#     # bound_vector = bond_vector_f(np.arange(iters+1), 2, error[0])
#     a1_min, a1_max, a2_min, a2_max = get_eigen_sym(A_tensor, funM)
#     l2 = plt.plot(error, c='y', label=f'$min(\lambda_1)$={a1_min:.0e}, $max(\lambda_1)$={a1_max:.0e}\n$min(\lambda_2)$={a2_min:.0e}, $max(\lambda_2)$={a2_max:.0e}')
#     # plt.plot(bound_vector, 'y--', label='bound')
#
#     A_tensor, B = generate_square_A(A_tall_hat_bad, 'DCT', M_random, X_true)
#     X, error = algo.CG_M_tensor(A_tensor, B, funM, invM, iters, X_true=X_true)
#     bound_vector = bond_vector_f(np.arange(iters+1), 2*10**4, error[0])
#
#     a1_min, a1_max, a2_min, a2_max = get_eigen_sym(A_tensor, funM)
#     l3 = plt.plot(error, c='purple', label=f'A obtained with DCT\n$min(\lambda_1)$={a1_min:.0e}, $max(\lambda_1)$={a1_max:.0e}\n$min(\lambda_2)$={a2_min:.0e}, $max(\lambda_2)$={a2_max:.0e}')
#
#     A_tensor, B = generate_square_A(A_tall_hat_good, 'DCT', M_random, X_true)
#     X, error = algo.CG_M_tensor(A_tensor, B, funM, invM, iters, X_true=X_true)
#     bound_vector = bond_vector_f(np.arange(iters+1), 2, error[0])
#     a1_min, a1_max, a2_min, a2_max = get_eigen_sym(A_tensor, funM)
#     l4 = plt.plot(error, c='orange', label=f'A obtained with DCT\n$min(\lambda_1)$={a1_min:.0e}, $max(\lambda_1)$={a1_max:.0e}\n$min(\lambda_2)$={a2_min:.0e}, $max(\lambda_2)$={a2_max:.0e}')
#
#     plt.xlabel('iterations')
#     plt.yscale("log")
#     plt.title(f'random state of M = {M_random}')
#     plt.ylabel('error')
#     plt.legend(loc='upper right')
# fig.legend([l1, l2, l3, l4], labels=line_labels, bbox_to_anchor=(0.8, 0.7))
# plt.suptitle('M prod conjugate gradient')
# plt.tight_layout()
# plt.savefig(path_to_save+'eigenvalues_experiment_CG_add_blendenpik')
# plt.show()

link = 'https://www.dropbox.com/scl/fo/vwn07zx7omd0zvqw0laxs/h?dl=0&rlkey=oyllju7bignxhr1liqdqh8b23'