import numpy as np 
import cvxpy as cp
import matplotlib.pyplot as plt
from data.sphere_fit_data import *

n = 2

c = cp.Variable(n)

t = cp.Variable(1)


squared_norms = np.sum(U**2, axis=0)

obj = cp.Minimize(cp.sum(cp.square(squared_norms - 2 * U.T @ c - t)))

problem = cp.Problem(obj)

problem.solve()

t = t.value
c = c.value

r = np.sqrt(t + np.sum(c**2))

print(f'optimal value: \n{problem.value}')
print(f'optimal values for c: \n{c}')
print(f'optimal value for t: \n{t}')
print(f'optimal value for r: \n{r}')

# Plot the points and the circle
plt.scatter(U[0], U[1], label='points', color='orange')
circle = plt.Circle(c, r, label='Circle', facecolor='none', edgecolor='blue')
plt.gca().add_patch(circle)
plt.axis('equal')
plt.grid()

plt.show()