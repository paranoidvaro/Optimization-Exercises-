import cvxpy as cp
import numpy as np

n = 2

x = cp.Variable(n)

A = np.array([[2., 1.], [1.,3.]])
b = np.array([1., 1.])
constraints = [A @ x >= b, x >= 0]

obj_a = cp.Minimize(x[0] + x[1]) # [0.4 0.2]

obj_b = cp.Minimize(-x[0] - x[1]) # None

obj_c = cp.Minimize(x[0]) # [-2.24914418e-10 ≈ 0   1.55371590e+00]

obj_d = cp.Minimize(cp.maximum(x[0],x[1])) # [0.33333333 0.33333333]

obj_e = cp.Minimize(x[0]**2 + 9*x[1]**2) # [0.5  0.16666667]

list = [obj_a, obj_b, obj_c, obj_d, obj_e]

results = np.zeros(5)
points = []

for index, element in enumerate(list):
    problem = cp.Problem(element, constraints) 
    result = problem.solve()
    points.append(x.value)
    results[index] = result
    
print(points)
print(results)







