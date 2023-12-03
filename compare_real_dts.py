import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import algo
import pickle
from matplotlib.pyplot import cm

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

X_pred = algo.Cholesky_direct(tenA_norm, omatB_norm)
plt.plot(X_pred[:, 0, :])
plt.show()
identity_error = algo.tensor_frob_norm(algo.facewise_mult(tenA_norm, X_pred)-omatB_norm)
print(identity_error)


from mprod import generate_haar, generate_dct

success_count = 0
fail_count = 0
for i in range(10**4):
       funM, invM = generate_haar(n, random_state=i)

       tenAhat = funM(tenA)
       tenAhat_norm = tenAhat - np.expand_dims(tenAhat.mean(0), 0)
       # tenAbias = np.concatenate((invM(np.ones((m, 1, n))), invM(tenA_norm_hat)), 1)

       omatBhat = funM(omatB)
       omatBhat_norm = (omatBhat - np.expand_dims(omatBhat.mean(0), 0))

       X_pred_hat = algo.Cholesky_direct(tenAhat_norm, omatBhat_norm)
       error_hat = algo.tensor_frob_norm(algo.facewise_mult(tenAhat_norm, X_pred_hat)-omatBhat_norm)
       # print(error_hat)
       error = algo.tensor_frob_norm(algo.m_prod(invM(tenAhat_norm), invM(X_pred_hat), funM, invM) - invM(omatBhat_norm))
       # print(error)
#
       if error < identity_error:
#               # print(i, error, 'SUCCESS')
#               success_count+=1
#        else:
#               print(identity_error-error_hat)
#               fail_count+=1
# print(success_count/10**4, fail_count/10**4)

              plt.figure(figsize=(15, 5))
              plt.subplot(121)
              plt.plot(X_pred[:, 0, :])
              plt.title('orig space')

              plt.subplot(122)
              plt.plot(funM(X_pred)[:, 0, :])
              plt.title('hat space')
              plt.suptitle(f'random state funM = {i}, error = {np.round(error, 1)}')
              plt.show()
       #