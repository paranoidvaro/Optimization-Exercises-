import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

n = 13

m = np.array([1, 5, 6, 15, 18, 20, 22, 11, 22, 8, 9, 4, 2])

p = cp.Variable(n)


obj = cp.Maximize(cp.sum(cp.multiply(m, cp.log(p))))

constraints = [cp.sum(p) == 1]

problem = cp.Problem(obj, constraints)

problem.solve()

print(f'the optimal values for p are: \n{p.value}')

#### now we compute the empirical p

m_sum = np.sum(m)

empirical_p = m / m_sum

print(f'the empirical values for p are: \n{empirical_p}')

x_s = np.array([i for i in range(n)])

plt.plot(x_s, p.value, color='green')
plt.show()

plt.plot(x_s, empirical_p, color='orange')
plt.show()


