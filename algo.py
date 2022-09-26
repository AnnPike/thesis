import numpy as np
from mprod import  m_prod


def normalize(X, funM, invM, tol=10**-10):
    V = funM(X)
    m, p, n = X.shape #p==1
    a_all = np.ones(n)
    for i in range(n):
        a = np.linalg.norm(V[:, :, i].squeeze(), None)
        # print(a)
        a_all[i] = a
        if a>tol:
            V[:, :, i] = V[:, :, i]/a
        else:
            print(a)
            V[:,:,i] = np.random.randn(m, 1)
            a = np.linalg.norm(V[:, :, i].squeeze(), None)
            V[:, :, i] = V[:, :, i]/a
            a_all[i] = 0
    V = invM(V)
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

        alpha = m_prod(alpha_num,inverse_of_tube(alpha_den, funM, invM),funM, invM)
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