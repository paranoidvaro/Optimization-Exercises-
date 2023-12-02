import numpy as np
import cvxpy as cp 
import matplotlib.pyplot as plt


assets = 500

scenarios = 20



z = cp.Variable(1)
x = cp.Variable(1)
y = cp.Variable(1)



# print(matrix.T @ x)

obj = cp.Minimize(cp.sum(cp.square(x + z + y)))


constraints = [x + z <= 1 + cp.geo_mean(cp.vstack([x - cp.quad_over_lin(z,y), y]))]

problem = cp.Problem(obj, constraints)



problem.solve()






print(z.value)
print(x.value)
print(y.value)
