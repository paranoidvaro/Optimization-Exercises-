import numpy as np
import cvxpy as cp 
import matplotlib.pyplot as plt 
from data.blend_design import *


λ = cp.Variable(k)
# print(λ.shape)

# print(W.shape)

constraints = [
    np.log(P).T @ λ <= np.log(P_spec),
    np.log(D).T @ λ <= np.log(D_spec),
    np.log(A).T @ λ <= np.log(A_spec),
    cp.sum(λ) == 1,
    λ >= 0]

obj = cp.Minimize(0)
prob = cp.Problem(obj, constraints)
prob.solve()

w_des = np.log(W) @ λ.value

w = np.exp(w_des)

print(f'W: {w}')
print(f'optimal λ: {λ.value}')