import numpy as np

def generate_incoherent_matrix(m, p):
    mat = np.random.rand(m, p)+10e-8*np.ones((m, p))
    return mat


def generate_coherent_matrix(m, p):
    mat = np.concatenate((np.diag(np.random.rand(p)), np.zeros((m-p, p))))+10e-8*np.ones((m, p))
    np.random.shuffle(mat)
    return mat


def generate_semi_coherent_matrix(m, p):
    B = np.random.rand(m - p // 2, p // 2)
    A1 = np.concatenate((B, np.zeros((m - p // 2, p-p // 2))), 1)
    A2 = np.concatenate((np.zeros((p // 2, p-p // 2)), np.diag(np.ones(p // 2))), 1)
    mat = np.concatenate((A1, A2)) + 10e-8 * np.ones((m, p))
    np.random.shuffle(mat)
    return mat


def generate_tensor(m, p, n, coh):
    if coh == 'incoherent':
        gen_mat_fun = generate_incoherent_matrix
    elif coh == 'coherent':
        gen_mat_fun = generate_coherent_matrix
    elif coh == 'semi-coherent':
        gen_mat_fun = generate_semi_coherent_matrix
    synth_tensor = np.empty((m, p, 0))
    for i in range(n):
        slice_i = np.expand_dims(gen_mat_fun(m, p), -1)
        synth_tensor = np.concatenate((synth_tensor, slice_i), 2)
    return synth_tensor
