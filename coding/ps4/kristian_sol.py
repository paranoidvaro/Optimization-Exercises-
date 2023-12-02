import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from data.sphere_fit_data import *

# Variables
c = cp.Variable(n)
t = cp.Variable()

# Compute squared norms of each column of U_matrix
squared_norms = np.sum(U**2, axis=0)



# Objective function
deviations = cp.power(squared_norms - 2 * U.T @ c - t, 2)
objective = cp.Minimize(cp.sum(deviations))

# Solve the optimization problem
prob = cp.Problem(objective)
prob.solve()

# Display results
print(f'status: {prob.status}')
print(f'optimal value: {prob.value}')
print(f'optimal c var: {c.value}')
print(f'optimal t var: {t.value}')

r = cp.sqrt(t + cp.power(cp.norm(c), 2)).value
print(f'optimal r var: {r}')

# Plot the points and the circle
plt.scatter(U[0], U[1], label='points')
circle = plt.Circle(c.value, r, label='Circle', facecolor='none', edgecolor='blue')
plt.gca().add_patch(circle)
plt.legend()
plt.grid(True)
plt.show()