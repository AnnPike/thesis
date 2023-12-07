import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import algo
import pickle
from matplotlib.pyplot import cm
from mprod import generate_haar, generate_dct, x_m3
from numpy import linalg as LA


R = 1
tenX = pickle.load(open('datasets/rainfall_ten', 'rb'))
print(tenX.shape)

tenA = tenX[:, :-2, :]
m, p, n = tenA.shape
omatB = np.expand_dims(tenX[:, -2, :], 1)

colors = cm.rainbow(np.linspace(0, 1, n))
for i in range(n):
       plt.plot(tenA[-1, :, i], c=colors[i])
       plt.scatter(p, omatB[-1, :, i], c=colors[i])
plt.show()
print(tenA.shape, omatB.shape)

tenA_norm = (tenA - np.expand_dims(tenA.mean(0), 0))
omatB_norm = (omatB - np.expand_dims(omatB.mean(0), 0))
tenAbias = np.concatenate((np.ones((m, 1, n)), tenA_norm), 1)

X_pred_identity = algo.Cholesky_direct(tenA_norm, omatB_norm, reg=R)
identity_error = algo.tensor_frob_norm(algo.facewise_mult(tenA_norm, X_pred_identity)-omatB_norm)
print(identity_error)


plt.figure(figsize=(15, 10))
plt.subplot(3, 2, 1)
plt.plot(X_pred_identity[:, 0, :].T)
plt.title(f'original solution, residual = {np.round(identity_error, 2)}')



funM, invM = generate_dct(n)
# tenAhat_norm = funM(tenA_norm)
# omatBhat_norm = funM(omatB_norm)
X_pred = algo.Cholesky_direct(tenA_norm, omatB_norm, reg=R, funM=funM, invM=invM)
error_hat = algo.tensor_frob_norm(algo.facewise_mult(funM(tenA_norm), funM(X_pred))-funM(omatB_norm))
error = algo.tensor_frob_norm(algo.m_prod(tenA_norm, X_pred, funM, invM) - omatB_norm)

plt.subplot(3, 2, 3)
plt.plot(invM(X_pred)[:, 0, :].T)
plt.title(f'orig space, residual = {np.round(error, 2)}')
plt.subplot(3, 2, 4)
plt.plot(funM(X_pred)[:, 0, :].T)
plt.title(f'hat space, M = DCT')

def generate_corr(ten):
    m, p, n = ten.shape
    X = ten.sum(1)
    X_cent = (X - X.mean(0))/X.std(0)
    X_corr = X_cent.T @ X_cent/m
    eigenvalues, eigenvectors = LA.eig(X_corr)
    return x_m3(eigenvectors.T), x_m3(eigenvectors)


funM, invM = generate_corr(tenA)

X_pred = algo.Cholesky_direct(tenA_norm, omatB_norm, reg=R, funM=funM, invM=invM)
error_hat = algo.tensor_frob_norm(algo.facewise_mult(funM(tenA_norm), funM(X_pred))-funM(omatB_norm))
error = algo.tensor_frob_norm(algo.m_prod(tenA_norm, X_pred, funM, invM) - omatB_norm)
print(np.isclose(error_hat, error))

plt.subplot(3, 2, 5)
plt.plot(invM(X_pred)[:, 0, :].T)
plt.title(f'orig space, residual = {np.round(error, 2)}')
plt.subplot(3, 2, 6)
plt.plot(funM(X_pred)[:, 0, :].T)
plt.title(f'hat space, M = eigenvectors of cov mat')

plt.tight_layout()
plt.savefig('datasets/rainfall_figure')
plt.show()

