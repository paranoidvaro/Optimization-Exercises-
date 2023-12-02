import numpy as np
import cvxpy as cp
from data.cens_fit_data import *

X = X.reshape(20, 100)

c = cp.Variable((n,1))

s = cp.Variable((K - M, 1))

constraints = [s >= D]


X_1 = X[:, :M].T

X_2 = X[:, M:].T


obj = cp.Minimize(cp.sum_squares(y - X_1 @ c) + cp.sum_squares(s - X_2 @ c))

problem = cp.Problem(obj, constraints)

problem.solve()

c_hat = c.value
print(f'c_hat is:\n {c_hat}')

# finding c_ls

c_ls = cp.Variable((n, 1))

residual_1 = y - X_1 @ c_ls
obj = cp.Minimize(cp.norm2(residual_1))
problem = cp.Problem(obj)

problem.solve()

c_ls_hat = c_ls.value

print(f'c_ls_hat is:\n {c_ls_hat}')

relative_errors = ((np.linalg.norm(c_true - c_hat)) / np.linalg.norm(c_true),  (np.linalg.norm(c_true - c_ls_hat)) / np.linalg.norm(c_true))

print(f'The relative error using the censored data is: {relative_errors[0]} \n instead just ignoring the data we obtain: {relative_errors[1]}')




