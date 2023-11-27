import algo
import random_generation as rg
import numpy as np
from mprod import generate_haar, generate_dct, m_prod

m, p, n = 100, 10, 5

tenA_hat = rg.generate_tensor(m, p, n, 'coherent')
omatX_hat = np.random.randn(p, 1, n)

# omatNoise = np.random.randn(m, 1, n)
omatNoise = np.zeros((m, 1, n))
omatB_hat = algo.facewise_mult(tenA_hat, omatX_hat) + omatNoise


X_sol_hat = algo.Cholesky_direct(tenA_hat, omatB_hat)
print(algo.tensor_frob_norm(X_sol_hat-omatX_hat))

funM, invM = generate_haar(n, random_state=2)
print(algo.tensor_frob_norm(invM(X_sol_hat)-invM(omatX_hat)))
