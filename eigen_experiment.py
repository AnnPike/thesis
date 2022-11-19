import numpy as np
from mprod import  m_prod
from mprod import  generate_haar, generate_dct
import algo
from scipy.stats import ortho_group
import helper_plot
import matplotlib.pyplot as plt
import time


def energy_norm(E, A_sym, funM, invM):
    E_hat = funM(E)
    A_hat = funM(A_sym)
    nu_hat_sq = np.einsum('mpi,pli->mli', np.einsum('mpi,pli->mli', E_hat.transpose(1, 0, 2), A_hat), E_hat)
    # print((nu_hat_sq<0).sum())
    return nu_hat_sq.sum()

# funM, invM = generate_haar(3, random_state=1)
# E = np.random.randn(5, 10, 3)
# A = np.random.rand(5, 5, 3)
#
# print(energy_norm(E, A, funM, invM))


def bound(i, k, E0_norm):
    return 2*E0_norm*((np.sqrt(k)-1)/(np.sqrt(k)+1))**i


bond_vector_f = np.vectorize(bound)
def nu_tensor_norm(E, A_tensor, funM, invM):
    return algo.tensor_frob_norm(algo.m_prod_three(E.transpose(1, 0, 2), A_tensor, E, funM, invM))


def generate_tall_matrix(m, p, eigenmin, k):
    H = ortho_group.rvs(dim=m)
    low = np.sqrt(eigenmin)
    high = np.sqrt(eigenmin*k)
    eigen = np.random.uniform(low, high, p)
    LAM = np.zeros((p, p))
    np.fill_diagonal(LAM, eigen)
    P = np.matmul(H[:, :p], LAM)
    Q = ortho_group.rvs(dim=p)
    A = np.matmul(P, Q.T)
    return A


def generate_tall_A_hat(m, p, eigen1, eigen2, k):

    A1 = generate_tall_matrix(m, p, eigen1, k)
    A2 = generate_tall_matrix(m, p, eigen2, k)

    A_hat = np.concatenate([A1.reshape(m, p, 1), A2.reshape(m, p, 1)], 2)
    return A_hat


def generate_tall_A(A_hat, transform, funM, invM, X_true, B):

    if transform == 'DCT':
        fun_dct, inv_dct = generate_dct(2)
        A_tall_tensor = inv_dct(A_hat)
    else:
        A_tall_tensor = invM(A_hat)
    if X_true is None:
        return A_tall_tensor, B
    else:
        B = m_prod(A_tall_tensor, X_true, funM, invM)
    return A_tall_tensor, B






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

def get_eigen(A, funM, sym_input):
    A_hat = funM(A)
    if not sym_input:
        A_hat_sym = np.einsum('mpi,pli->mli', A_hat.transpose(1, 0, 2), A_hat)
    else:
        A_hat_sym = A_hat.copy()
    lambda1 = np.linalg.eigvals(A_hat_sym[:, :, 0])
    a1_min = np.round(lambda1.min(), 5)
    a1_max = np.round(lambda1.max(), 5)
    lambda2 = np.linalg.eigvals(A_hat_sym[:, :, 1])
    a2_min = np.round(lambda2.min(), 5)
    a2_max = np.round(lambda2.max(), 5)

    return a1_min, a1_max, a2_min, a2_max


def calculate_cond(A, funM, sym_input):
    a1_min, a1_max, a2_min, a2_max = get_eigen(A, funM, sym_input)
    k1 = a1_max/a1_min
    k2 = a2_max/a2_min
    return int(max(k1, k2))

def fill_dict_lines(dict_of_lines, A_tensor, B, funM, invM, error, list_of_X, plot_what, matrix_type):
    if plot_what == 'error':
        dict_of_lines[i][matrix_type] = error
    if plot_what == 'normalized error':
        dict_of_lines[i][matrix_type] = error / error[0]
    if plot_what == 'residual':
        list_of_res = res_norm_steps(A_tensor, list_of_X, B, funM, invM)
        dict_of_lines[i][matrix_type] = list_of_res
    if plot_what == 'normalized residual':
        list_of_res = norm_res_norm_steps(A_tensor, list_of_X, B, funM, invM)
        dict_of_lines[i][matrix_type] = list_of_res
    return dict_of_lines


path_to_save = '/home/anna/uni/thesis/numerical_results/eigen_n2_numerical/'
np.random.seed(1)

iters = 10
M_list = ['DCT', 21, 127, 333]

# define parameters
m, p = 500, 50
k = 100
degree = 6

random_X = np.random.randn(p, 1, 2)

start_time = time.time()
X_true = random_X
s = p*6

np.random.seed(1)
A_tall_hat_bad = generate_tall_A_hat(m, p, eigen1=1, eigen2=10**degree, k=k)
np.random.seed(1)
A_tall_hat_good = generate_tall_A_hat(m, p, eigen1=1, eigen2=1, k=k)

plt.figure(figsize=(20,10))
for i in range(4):
    plt.subplot(2, 2, i+1)
    M_random = M_list[i]

    if M_random == 'DCT':
        funM, invM = generate_dct(2)
    else:
        funM, invM = generate_haar(2, random_state=M_random)

    A_tensor, B = generate_tall_A(A_tall_hat_bad, 'original M', funM, invM, X_true, None)
    list_of_X = algo.LSQR_mprod_tuples(A_tensor, B, funM, invM, iters)
    list_of_E = [list_of_X[i]-X_true for i in range(iters+1)]
    As = m_prod(A_tensor.transpose(1, 0, 2), A_tensor, funM, invM)
    k = calculate_cond(As, funM, True)
    error_vector = np.array([energy_norm(E, As, funM, invM) for E in list_of_E])
    bound_vector = bond_vector_f(np.arange(iters+1), k, error_vector[0])

    # plt.plot(error_vector, 'b')
    # plt.plot(bound_vector, '--b')
    #preconditioning

    P, R = algo.blendenpick(A_tensor, funM, invM, s=s)

    list_of_X = algo.LSQR_mprod_tuples_precond(A_tensor, B, R, funM, invM, iters)
    list_of_E = [list_of_X[i] - X_true for i in range(iters + 1)]
    A_tensor_precond = m_prod(A_tensor, R, funM, invM)
    Asp = m_prod(A_tensor_precond.transpose(1, 0, 2), A_tensor_precond, funM, invM)
    k = calculate_cond(Asp, funM, True)
    error_vector = np.array([energy_norm(E, Asp, funM, invM) for E in list_of_E])
    error_vector_s = np.array([energy_norm(E, As, funM, invM) for E in list_of_E])
    bound_vector = bond_vector_f(np.arange(iters + 1), k, error_vector[0])
    bound_vector_s = bond_vector_f(np.arange(iters + 1), k, error_vector_s[0])

    # plt.plot(error_vector/error_vector[0], 'purple', alpha=0.5,label='error energy norm Asp')
    # plt.plot(bound_vector/error_vector[0], 'purple', alpha=0.5,ls='--', label='bound Asp')
    # plt.plot(error_vector_s/error_vector_s[0], 'purple',  label='error energy norm As')
    # plt.plot(bound_vector_s/error_vector_s[0], 'purple', ls='--', label='bound As')

    A_tensor, B = generate_tall_A(A_tall_hat_good, 'original M', funM, invM, X_true, None)
    list_of_X = algo.LSQR_mprod_tuples(A_tensor, B, funM, invM, iters)
    list_of_E = [list_of_X[i] - X_true for i in range(iters + 1)]
    As = m_prod(A_tensor.transpose(1, 0, 2), A_tensor, funM, invM)
    k = calculate_cond(As, funM, True)
    error_vector = np.array([energy_norm(E, As, funM, invM) for E in list_of_E])
    bound_vector = bond_vector_f(np.arange(iters + 1), k, error_vector[0])

    # plt.plot(error_vector, 'y')
    # plt.plot(bound_vector, 'y', ls='--')

    P, R = algo.blendenpick(A_tensor, funM, invM, s=s)
    list_of_X = algo.LSQR_mprod_tuples_precond(A_tensor, B, R, funM, invM, iters)
    list_of_E = [list_of_X[i] - X_true for i in range(iters + 1)]
    A_tensor_precond = m_prod(A_tensor, R, funM, invM)
    Asp = m_prod(A_tensor_precond.transpose(1, 0, 2), A_tensor_precond, funM, invM)
    k = calculate_cond(Asp, funM, True)
    error_vector = np.array([energy_norm(E, Asp, funM, invM) for E in list_of_E])
    error_vector_s = np.array([energy_norm(E, As, funM, invM) for E in list_of_E])
    bound_vector = bond_vector_f(np.arange(iters + 1), k, error_vector[0])
    bound_vector_s = bond_vector_f(np.arange(iters + 1), k, error_vector_s[0])

    plt.plot(error_vector/error_vector[0], 'orange', alpha=0.5, label= 'energy norm Asp')
    plt.plot(bound_vector/error_vector[0], 'orange', alpha=0.5, ls='--', label='bound Asp')
    plt.plot(error_vector_s/error_vector_s[0], 'orange', label='energy norm As')
    plt.plot(bound_vector_s/error_vector_s[0], 'orange', ls='--', label='bound As')
    plt.legend()
    plt.ylabel('normalized error')

    plt.yscale('log')
plt.savefig(path_to_save+'bound_log')
plt.show()


