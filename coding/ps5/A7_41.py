import cvxpy as cp 
import numpy as np 

from data.poisson_ar_data import *

a = cp.Variable()
b = cp.Variable()

x = np.hstack([np.array([0]), x])

obj = 0

for t in range(num_points):
    obj += - cp.exp(a + b * x[t])
    obj += x[t + 1] * a
    obj += x[t + 1] * x[t] * b




problem = cp.Problem(cp.Maximize(obj))

problem.solve()

print(f'Problem Status: {problem.status}')
print(f'The Maximum Likelihood Estimate for ν is: {np.exp(a.value)}\n for ω is: {np.exp(b.value)}')