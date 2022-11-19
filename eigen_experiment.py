import numpy as np
from mprod import  m_prod
from mprod import  generate_haar, generate_dct
import algo
from scipy.stats import ortho_group
import helper_plot
import matplotlib.pyplot as plt
import time

def bound(i, k, E0):
    return 2*E0*((np.sqrt(k)-1)/(np.sqrt(k)+1))**i

bond_vector_f = np.vectorize(bound)

def energy_norm(E, A_sym, funM):
    E_hat = funM(E)
    A_hat = funM(A_sym)
    nu_hat_sq = np.einsum('mpi,pli->mli', np.einsum('mpi,pli->mli', E_hat.transpose(1, 0, 2), A_hat), E_hat)
    # print((nu_hat_sq<0).sum())
    return nu_hat_sq.sum()

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


def generateB_noisy_X(tensorA, X_true, sigma, funM, invM):
    m, p, n = X_true.shape
    X_distorded = X_true + sigma*np.random.randn(m, p, n)
    B_noisy = m_prod(tensorA, X_distorded, funM, invM)
    return B_noisy


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


def get_eigen(A_hat):
    lambda1 = np.linalg.eigvals(A_hat[:, :, 0])
    a1_min = np.round(lambda1.min(), 5)
    a1_max = np.round(lambda1.max(), 5)
    lambda2 = np.linalg.eigvals(A_hat[:, :, 1])
    a2_min = np.round(lambda2.min(), 5)
    a2_max = np.round(lambda2.max(), 5)

    return a1_min, a1_max, a2_min, a2_max


def calculate_cond(A, funM):
    A_hat = funM(A)
    a1_min, a1_max, a2_min, a2_max = get_eigen(A_hat)
    k1 = a1_max/a1_min
    k2 = a2_max/a2_min
    return int(max(k1, k2))


def fill_dict_lines(dict_of_lines, A_tensor, B, funM, invM, error, list_of_X, plot_what, matrix_type, sigma):
    if plot_what == 'error':
        dict_of_lines[i][sigma][matrix_type] = error
    if plot_what == 'normalized error':
        dict_of_lines[i][sigma][matrix_type] = error / error[0]
    if plot_what == 'residual':
        list_of_res = res_norm_steps(A_tensor, list_of_X, B, funM, invM)
        dict_of_lines[i][sigma][matrix_type] = list_of_res
    if plot_what == 'normalized residual':
        list_of_res = norm_res_norm_steps(A_tensor, list_of_X, B, funM, invM)
        dict_of_lines[i][sigma][matrix_type] = list_of_res
    return dict_of_lines


path_to_save = '/home/anna/uni/thesis/numerical_results/eigen_n2_numerical/'
np.random.seed(1)

iters = 40
M_list = ['DCT', 21, 127, 333]

# define parameters
m, p = 500, 50
k = 100
degree = 6

X_true = np.random.randn(p, 1, 2)
B = None
plot_what_options = ['residual', 'error', 'normalized residual', 'normalized error']


np.random.seed(1)
A_tall_hat_bad = generate_tall_A_hat(m, p, eigen1=1, eigen2=10**degree, k=k)
np.random.seed(1)
A_tall_hat_good = generate_tall_A_hat(m, p, eigen1=1, eigen2=1, k=k)


# genrating vectors to plot

for plot_what in plot_what_options:
    globals()[f'dict_of_lines_{plot_what}'] = {}
dict_of_cond = {}

sigma_list = [0.01, 1]

for i in range(4):

    M_random = M_list[i]
    dict_of_cond[i] = {}
    for plot_what in plot_what_options:
        globals()[f'dict_of_lines_{plot_what}'][i] = {}
    if M_random == 'DCT':
        funM, invM = generate_dct(2)
    else:
        funM, invM = generate_haar(2, random_state=M_random)
    # A_tensor_bad = invM(A_tall_hat_bad)
    # A_tensor_good = invM(A_tall_hat_good)

    A_tensor, B_true = generate_tall_A(A_tall_hat_bad, 'original M', funM, invM, X_true, None)
    A_sym = m_prod(A_tensor.transpose(1, 0, 2), A_tensor, funM, invM)
    dict_of_cond[i]['orig bad'] = calculate_cond(A_sym, funM)
    for sigma in sigma_list:
        B = generateB_noisy_X(A_tensor, X_true, sigma, funM, invM)
        list_of_X = algo.LSQR_mprod_tuples(A_tensor, B, funM, invM, iters)
        list_of_E = [X - X_true for X in list_of_X]
        error = np.array([energy_norm(E, A_sym, funM) for E in list_of_E])
        for plot_what in plot_what_options:
            globals()[f'dict_of_lines_{plot_what}'][i][sigma] = {}
            fill_dict_lines(globals()[f'dict_of_lines_{plot_what}'], A_tensor, B, funM, invM, error, list_of_X, plot_what, 'orig bad', sigma)

    #preconditioning
    s = p*6

    P, R = algo.blendenpick(A_tensor, funM, invM, s=s)
    A_tensor_precond = m_prod(A_tensor, R, funM, invM)
    Ap_sym = m_prod(A_tensor_precond.transpose(1, 0, 2), A_tensor_precond, funM, invM)
    dict_of_cond[i]['prec bad'] = calculate_cond(Ap_sym, funM)
    for sigma in sigma_list:
        B = generateB_noisy_X(A_tensor, X_true, sigma, funM, invM)
        list_of_X = algo.LSQR_mprod_tuples_precond(A_tensor, B, R, funM, invM, iters)
        list_of_E = [X - X_true for X in list_of_X]
        error = np.array([energy_norm(E, A_sym, funM) for E in list_of_E])
        for plot_what in plot_what_options:
            fill_dict_lines(globals()[f'dict_of_lines_{plot_what}'], A_tensor, B, funM, invM, error, list_of_X, plot_what,
                            'prec bad', sigma)

    A_tensor, B_true = generate_tall_A(A_tall_hat_good, 'original M', funM, invM, X_true, B)
    A_sym = m_prod(A_tensor.transpose(1, 0, 2), A_tensor, funM, invM)
    dict_of_cond[i]['orig good'] = calculate_cond(A_sym, funM)
    for sigma in sigma_list:
        B = generateB_noisy_X(A_tensor, X_true, sigma, funM, invM)
        list_of_X = algo.LSQR_mprod_tuples(A_tensor, B, funM, invM, iters)
        list_of_E = [X - X_true for X in list_of_X]
        error = np.array([energy_norm(E, A_sym, funM) for E in list_of_E])
        for plot_what in plot_what_options:
            fill_dict_lines(globals()[f'dict_of_lines_{plot_what}'], A_tensor, B, funM, invM, error, list_of_X, plot_what, 'orig good', sigma)


    P, R = algo.blendenpick(A_tensor, funM, invM, s=s)
    A_tensor_precond = m_prod(A_tensor, R, funM, invM)
    Ap_sym = m_prod(A_tensor_precond.transpose(1, 0, 2), A_tensor_precond, funM, invM)
    dict_of_cond[i]['prec good'] = calculate_cond(Ap_sym, funM)
    for sigma in sigma_list:
        B = generateB_noisy_X(A_tensor, X_true, sigma, funM, invM)
        list_of_X = algo.LSQR_mprod_tuples_precond(A_tensor, B, R, funM, invM, iters)
        list_of_E = [X - X_true for X in list_of_X]
        error = np.array([energy_norm(E, A_sym, funM) for E in list_of_E])
        for plot_what in plot_what_options:
            fill_dict_lines(globals()[f'dict_of_lines_{plot_what}'], A_tensor, B, funM, invM, error, list_of_X, plot_what,
                            'prec good', sigma)


for plot_what in plot_what_options:
    helper_plot.plot_4M_2A_precond(M_list, globals()[f'dict_of_lines_{plot_what}'], plot_what, degree, m, p, s, dict_of_cond)
    name = f'eigenvalues_experiment_LSQR_blendenpick_{m}_{p}_s{s}_k{k}_{plot_what.replace(" ", "_")}_X_true_B_noisyX'
    plt.savefig(path_to_save + name)
    plt.show()




