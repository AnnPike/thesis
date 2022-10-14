import numpy as np
from mprod import  m_prod


def normalize(X, funM, invM, tol=10**-10):
    V_hat = funM(X)
    m, p, n = X.shape #p==1
    a_all = np.ones(n)
    for i in range(n):
        a = np.linalg.norm(V_hat[:, :, i].squeeze(), None)
        # print(a)
        a_all[i] = a
        if a > tol:
            V_hat[:, :, i] = V_hat[:, :, i]/a
        else:
            print(a)
            V_hat[:, :, i] = np.random.randn(m, 1)
            a = np.linalg.norm(V_hat[:, :, i].squeeze(), None)
            V_hat[:, :, i] = V_hat[:, :, i]/a
            a_all[i] = 0
    V = invM(V_hat)
    a_all = invM(a_all.reshape((1, 1, -1)))
    return V, a_all


def inverse_of_tube(a, funM, invM):
    n = a.shape[2]
    I_hat = np.ones((1, 1, n))
    a_hat = funM(a)
    a_inv_hat = np.divide(I_hat, a_hat)
    a_inv = invM(a_inv_hat)
    return a_inv



def m_prod_three(A, B, C, fun_m, inv_m):
    a_hat = fun_m(A)
    b_hat = fun_m(B)
    c_hat = fun_m(C)
    mult_hat = np.einsum('mpi,pli->mli', np.einsum('mpi,pli->mli', a_hat, b_hat), c_hat)
    return inv_m(mult_hat)


def tensor_frob_norm(A):
    return np.sqrt((A**2).sum())


def nu_tensor_norm(E, A_tensor, funM, invM):
    return tensor_frob_norm(m_prod_three(E.transpose(1, 0, 2), A_tensor, E, funM, invM))


def CG_M_tensor(A_tensor, B, funM, invM, iters, tol=10**-10, X_true=None):
    m, p, n = A_tensor.shape
    if m!=p:
        print('error: the frontal slices are not square matrices')
        return None
    X_zero = np.zeros((p, 1, n))

    R, a_all = normalize(B, funM, invM, tol)
    D = R.copy()
    X = X_zero.copy()
    if X_true is not None:
        E0 = X - X_true
        error_each_step = [nu_tensor_norm(E0, A_tensor, funM, invM)]
    for i in range(iters):
        alpha_num = m_prod(R.transpose((1,0,2)), R, funM, invM)
        alpha_den = m_prod_three(D.transpose((1,0,2)),A_tensor, D, funM, invM)

        alpha = m_prod(alpha_num, inverse_of_tube(alpha_den, funM, invM), funM, invM)
        X = X+m_prod(D, alpha, funM, invM)
        R_next = R-m_prod_three(A_tensor, D, alpha, funM, invM)
        beta_num = m_prod(R_next.transpose((1,0,2)), R_next, funM, invM)
        beta_den = m_prod(R.transpose((1,0,2)), R, funM, invM)
        beta = m_prod(inverse_of_tube(beta_den, funM, invM), beta_num,funM, invM)
        D = R_next+m_prod(D, beta, funM, invM)
        # print('D', D[0])
        R = R_next.copy()
        if X_true is not None:
            X_new = m_prod(X, a_all, funM, invM)
            # X_new = X.copy()
            E = X_new-X_true
            error_each_step.append(nu_tensor_norm(E, A_tensor, funM, invM))
    if X_true is not None:
        return X_new, error_each_step


def LSQR_mprod(A, C, funM, invM, itermax=25, tol=10 ** -5, X_true=None):
    ### A - m*p*n, C - m*1*n ###
    # initialization
    A_tensor = m_prod(np.transpose(A, (1, 0, 2)), A, funM, invM)

    X = np.zeros((A.shape[1], C.shape[1], C.shape[2]))  # X - l*s*p
    beta = tensor_frob_norm(C)
    U = C / beta

    V_wave = m_prod(np.transpose(A, (1, 0, 2)), U, funM, invM)
    alpha = tensor_frob_norm(V_wave)
    V = V_wave / alpha  # l*s*p

    W = V.copy()
    ro_ = alpha
    fi_ = beta
    if X_true is not None:
        E0 = X - X_true
        error_each_step = [nu_tensor_norm(E0, A_tensor, funM, invM)]
    for i in range(itermax):
        # bidiagonalization
        U_wave = m_prod(A, V, funM, invM) - alpha * U
        beta = tensor_frob_norm(U_wave)
        U = U_wave / beta

        V_wave = m_prod(np.transpose(A, (1, 0, 2)), U, funM, invM) - beta * V
        alpha = tensor_frob_norm(V_wave)
        V = V_wave / alpha

        # orthogonal transformation
        ro = np.sqrt(ro_ ** 2 + beta ** 2)
        c = ro_ / ro
        s = beta / ro

        theta = s * alpha
        ro_ = c * alpha
        fi = c * fi_
        fi_ = -s * fi_
        X = X + fi * W / ro
        W = V - theta * W / ro

        # if abs(fi_) < tol:
        #     break
        if X_true is not None:
            E = X - X_true
            error_each_step.append(nu_tensor_norm(E, A_tensor, funM, invM))

    return X, error_each_step


def LSQR_mprod_tuples(A, C, funM, invM, itermax=25, tol=10 ** -5, X_true=None):
    ### A - m*p*n, C - m*1*n ###
    # initialization
    def m_prod_fun(X, Y, funM=funM, invM=invM):
        return m_prod(X, Y, funM, invM)
    def m_prod_three_fun(X, Y, Z, funM=funM, invM=invM):
        return m_prod_three(X, Y, Z, funM, invM)
    A_tensor = m_prod_fun(np.transpose(A, (1, 0, 2)), A)
    list_of_X = []
    X = np.zeros((A.shape[1], C.shape[1], C.shape[2]))  # X - l*s*p
    list_of_X.append(X)
    U, beta = normalize(C, funM, invM)

    V_wave = m_prod_fun(np.transpose(A, (1, 0, 2)), U)
    V, alpha = normalize(V_wave, funM, invM) # V - l*s*p

    W = V.copy()
    ro_ = alpha.copy()
    fi_ = beta.copy()

    if X_true is not None:

        E0 = X - X_true
        error_each_step = [nu_tensor_norm(E0, A_tensor, funM, invM)]
    for i in range(itermax):
        # bidiagonalization
        U_wave = m_prod_fun(A, V) - m_prod_fun(U, alpha)
        U, beta = normalize(U_wave, funM, invM)

        V_wave = m_prod_fun(np.transpose(A, (1, 0, 2)), U) - m_prod_fun(V, beta)
        V, alpha = normalize(V_wave, funM, invM) # V - l*s*p

        # orthogonal transformation
        ro = invM(np.sqrt(funM(ro_) ** 2 + funM(beta) ** 2))
        # ro = np.sqrt(ro_ ** 2 + beta ** 2)
        ro_inv = inverse_of_tube(ro, funM, invM)
        c = m_prod_fun(ro_inv, ro_)
        s = m_prod_fun(ro_inv, beta)

        theta = m_prod_fun(s, alpha)
        ro_ = m_prod_fun(c, alpha)
        fi = m_prod_fun(c, fi_)
        fi_ = -m_prod_fun(s, fi_)
        X = X + m_prod_three_fun(W, ro_inv, fi)
        list_of_X.append(X)
        W = V - m_prod_three_fun(W, ro_inv, theta)

        # if abs(fi_) < tol:
        #     break
        if X_true is not None:
            E = X - X_true
            error_each_step.append(nu_tensor_norm(E, A_tensor, funM, invM))

    return X, error_each_step, list_of_X




def LSQR_mprod_tuples_precond(A, C, R, funM, invM, itermax=25, tol=10 ** -5, X_true=None):
    ### A - m*p*n, C - m*1*n ###
    # initialization
    def m_prod_fun(X, Y, funM=funM, invM=invM):
        return m_prod(X, Y, funM, invM)
    def m_prod_three_fun(X, Y, Z, funM=funM, invM=invM):
        return m_prod_three(X, Y, Z, funM, invM)
    A_tensor = m_prod_fun(A.transpose(1, 0, 2), A)

    Y = np.zeros((A.shape[1], C.shape[1], C.shape[2]))  # X - l*s*p
    list_of_X = []
    X = m_prod_fun(R, Y)
    list_of_X.append(X)
    U, beta = normalize(C, funM, invM)

    V_wave_p = m_prod_three_fun(R.transpose(1, 0, 2), A.transpose(1, 0, 2), U)
    V, alpha = normalize(V_wave_p, funM, invM) # V - l*s*p

    W = V.copy()
    ro_ = alpha.copy()
    fi_ = beta.copy()

    if X_true is not None:
        list_of_X.append(X)
        E0 = X - X_true
        error_each_step = [nu_tensor_norm(E0, A_tensor, funM, invM)]
    for i in range(itermax):
        # bidiagonalization

        U_wave = m_prod_three_fun(A, R, V) - m_prod_fun(U, alpha)
        U, beta = normalize(U_wave, funM, invM)

        V_wave = m_prod_three_fun(R.transpose(1,0,2), A.transpose(1,0,2), U) - m_prod_fun(V, beta)
        V, alpha = normalize(V_wave, funM, invM) # V - l*s*p

        # orthogonal transformation
        ro = invM(np.sqrt(funM(ro_) ** 2 + funM(beta) ** 2))
        # ro = np.sqrt(ro_ ** 2 + beta ** 2)
        ro_inv = inverse_of_tube(ro, funM, invM)
        c = m_prod_fun(ro_inv, ro_)
        s = m_prod_fun(ro_inv, beta)

        theta = m_prod_fun(s, alpha)
        ro_ = m_prod_fun(c, alpha)
        fi = m_prod_fun(c, fi_)
        fi_ = -m_prod_fun(s, fi_)
        Y = Y + m_prod_three_fun(W, ro_inv, fi)
        X = m_prod_fun(R, Y)
        list_of_X.append(X)
        W = V - m_prod_three_fun(W, ro_inv, theta)

        # if abs(fi_) < tol:
        #     break
        if X_true is not None:
            E = X - X_true
            error_each_step.append(nu_tensor_norm(E, A_tensor, funM, invM))

    return X, error_each_step, list_of_X


def inverse_tensor(tensor, funM, invM):
    tensor_hat = funM(tensor)
    m, p, n = tensor_hat.shape
    face_inv = np.concatenate([np.linalg.inv(tensor_hat[:, :, i]).reshape(m, p, 1) for i in range(n)], 2)
    tensor_inv = invM(face_inv)
    return tensor_inv

# from mprod import  generate_haar, generate_dct
# funM, invM = generate_haar(4, random_state=0)
# A = np.random.randn(5,5,4)
# print(A.shape)
# A_inv = inverse_tensor(A, funM, invM)
# C = m_prod(A, A_inv, funM, invM)
# D = funM(C)
# print(D)
# print('here')

def tensor_QR(tensor, funM, invM):
    m, p, n = tensor.shape
    tensor_hat = funM(tensor)
    tensor_Q = np.empty((m, p, 0))
    tensor_R = np.empty((p, p, 0))
    for i in range(n):
        Q, R = np.linalg.qr(tensor_hat[:, :, i], mode='reduced')

        tensor_Q = np.concatenate([tensor_Q, Q.reshape(m, p, 1)], 2)
        tensor_R = np.concatenate([tensor_R, R.reshape(p, p, 1)], 2)

    tensor_Q = invM(tensor_Q)
    tensor_R = invM(tensor_R)
    return tensor_Q, tensor_R



# from mprod import  generate_haar, generate_dct
# A = np.random.randn(5,3,4)
# funM, invM = generate_haar(4, random_state=0)
# tensor_Q, tensor_R = tensor_QR(A, funM, invM)
# A_nw = m_prod(tensor_Q, tensor_R, funM, invM)
# print('here')


def sampling_QR(tensor, funM, invM, s):
    m, p, n = tensor.shape
    sampling_tensor = np.random.randn(s, m, n)
    sampled_tensor = m_prod(sampling_tensor, tensor, funM, invM)
    tensor_Q, tensor_R = tensor_QR(sampled_tensor, funM, invM)
    tensor_precond = inverse_tensor(tensor_R, funM, invM)
    return tensor_R, tensor_precond


# from mprod import  generate_haar, generate_dct
# A = np.random.randn(100, 3, 4)
# funM, invM = generate_haar(4, random_state=0)
# A_hat = funM(A)
# C = np.einsum('mpi,pli->mli', A_hat.transpose(1, 0, 2), A_hat)
# all_eigen = np.empty((0))
# for i in range(4):
#     eig_i = np.linalg.eigvals(C[:, :, i])
#     all_eigen = np.concatenate([all_eigen, eig_i], 0)
# print(all_eigen)
# print(all_eigen.max()/all_eigen.min())
# R, tensor_precond = sampling_QR(A, funM, invM, 20)
#
# A_new = m_prod(A, tensor_precond, funM, invM)
# A_hat_new = funM(A_new)
# C_new = np.einsum('mpi,pli->mli', A_hat_new.transpose(1, 0, 2), A_hat_new)
# all_eigen = np.empty((0))
# for i in range(4):
#     eig_i = np.linalg.eigvals(C_new[:, :, i])
#     all_eigen = np.concatenate([all_eigen, eig_i], 0)
# print(all_eigen)
# print(all_eigen.max()/all_eigen.min())