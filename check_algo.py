import algo
import random_generation as rg
import numpy as np
from mprod import generate_haar, generate_dct, m_prod

m, p, n = 2000, 100, 300
funM, invM = generate_haar(n, 21)
np.random.seed(21)

tenA = rg.generate_tensor(m, p, n, 'coherent')
omatX = np.random.randn(p, 1, n)

# omatNoise = 0.1*np.random.randn(m, 1, n)
omatNoise = np.zeros((m, 1, n))
omatB = algo.m_prod(tenA, omatX, funM, invM) + omatNoise



# import time
# start_time = time.time()

X_sol_hat = algo.Cholesky_direct(funM(tenA), funM(omatB))
print(algo.tensor_frob_norm(invM(X_sol_hat)-omatX))
print(algo.tensor_frob_norm(X_sol_hat-funM(omatX)))

X_sol = algo.Cholesky_direct(tenA, omatB, funM=funM, invM=invM)

# print('it took:', time.time()-start_time)

print(algo.tensor_frob_norm(X_sol-omatX))
print(algo.tensor_frob_norm(funM(X_sol)-funM(omatX)))

