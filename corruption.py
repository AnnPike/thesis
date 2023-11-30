import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import algo


df = pd.read_csv('nat_and_corruption.csv')
df = df[['iso3c', 'year','corruption',
       'ideology_is_nationalist_percent_of_experts',
       'ideology_is_socialist_or_communist_percent_of_experts',
       'ideology_is_restorative_or_conservative_percent_of_experts',
       'ideology_is_separatist_or_autonomist_percent_of_experts',
       'ideology_is_religious_percent_of_experts']]

df = df[~df['corruption'].isna()]

c1 = pd.unique(df[df.year==2015].iso3c)
c2 = pd.unique(df[df.year==2012].iso3c)
c3 = pd.unique(df[df.year==2016].iso3c)
c = set(c1).intersection(c2).intersection(c3)
df = df[df['iso3c'].isin(c)]
df.fillna(0)

yX = df.iloc[:, 2:].to_numpy()
num_f = yX.shape[1]

YtenX = np.empty((165, num_f, 11))
for i in range(11):
    YtenX[:, :, i] = yX[np.arange(i, len(yX), 11)]
tenX = YtenX[:, 1:, :]
tenX[np.isnan(tenX)] = 0
omatY = np.expand_dims(tenX[:, 0, :], 1)


print(tenX.shape, omatY.shape)

# from sklearn.linear_model import RidgeCV as Tich
# lr = Tich(alphas=[1e-10, 0.1, 1, 5, 10])
#
#
# def solve_n_Ridge(tenA, omatB):
#     m, p, n = tenA.shape
#     X_pred = np.empty((p+1, n))
#     alpha_list = []
#     for i in range(n):
#         print(i)
#         lr.fit(tenA[:, :, i], omatB[:, :, i])
#         coef = lr.coef_
#         X_pred[:, i] = np.concatenate((lr.intercept_, coef.squeeze()))
#         alpha_list.append(lr.alpha_)
#     return X_pred, alpha_list
#
# X_pred, alpha_list = solve_n_Ridge(tenX, omatY)
#
# plt.plot(X_pred)
# plt.show()
# print(alpha_list)
# print(X_pred)


# tenX_normalized = tenX - np.expand_dims(tenX.mean(0), 0)
tenXbias = np.concatenate((np.ones((165, 1, 11)), tenX), 1)

X_pred = algo.Cholesky_direct(tenXbias, omatY)
plt.plot(X_pred[:, 0, :])
plt.show()
print(algo.tensor_frob_norm(algo.facewise_mult(tenXbias, X_pred)-omatY))


from mprod import generate_haar, generate_dct
for i in range(5):
       funM, invM = generate_haar(11, random_state=i+10)

       tenXbias = np.concatenate((invM(np.ones((165, 1, 11))), tenX), 1)
       X_pred = algo.Cholesky_direct(tenXbias, omatY, funM, invM)

       plt.figure(figsize=(15, 5))
       plt.subplot(121)
       plt.plot(X_pred[:, 0, :])
       plt.title('orig space')
       error = algo.tensor_frob_norm(algo.m_prod(tenXbias, X_pred, funM, invM)-omatY)
       plt.subplot(122)
       plt.plot(funM(X_pred)[:, 0, :])
       plt.title('hat space')
       plt.suptitle(f'random state funM = {i}, error = {np.round(error, 1)}')
       plt.show()
       print(error)
       print(algo.tensor_frob_norm(algo.facewise_mult(funM(tenXbias), funM(X_pred))-funM(omatY)))