import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import algo
import pickle
from matplotlib.pyplot import cm
from mprod import generate_haar, generate_dct


tenX = pickle.load(open('datasets/rainfall_ten', 'rb')).transpose(2, 1, 0)
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

X_pred_identity = algo.Cholesky_direct(tenA_norm, omatB_norm, reg=0.1)
identity_error = algo.tensor_frob_norm(algo.facewise_mult(tenA_norm, X_pred_identity)-omatB_norm)
print(identity_error)


plt.figure(figsize=(15, 10))
plt.subplot(3, 2, 1)
plt.plot(X_pred_identity[:, 0, :].T)
plt.title(f'original solution, error = {np.round(identity_error, 2)}')

found = 0
i = 0
while found < 2:
       funM, invM = generate_haar(n, random_state=i)
       # tenAhat_norm = funM(tenA_norm)
       # omatBhat_norm = funM(omatB_norm)
       X_pred = algo.Cholesky_direct(tenA_norm, omatB_norm, reg=0.1, funM=funM, invM=invM)
       error_hat = algo.tensor_frob_norm(algo.facewise_mult(funM(tenA_norm), funM(X_pred))-funM(omatB_norm))
       error = algo.tensor_frob_norm(algo.m_prod(tenA_norm, X_pred, funM, invM) - omatB_norm)
#
       if error < 0.99*identity_error:
           found+=1
           print(found)
           plt.subplot(3, 2, 2*found+1)
           plt.plot(invM(X_pred)[:, 0, :].T)
           plt.title(f'orig space, error = {np.round(error, 2)}')
           plt.subplot(3, 2, 2*found+2)
           plt.plot(funM(X_pred)[:, 0, :].T)
           plt.title(f'hat space, random state={i}')
       i += 1

plt.tight_layout()
plt.savefig('datasets/rainfall_figure_year')
plt.show()