import numpy as np
import cvxpy as cp


sigma = cp. Variable((4,4), symmetric = True)

x = np.array([[0.1], [0.2], [-0.05], [0.1]])

objective = cp.Maximize(x.T @ sigma @ x)

constraints = [
sigma [0][0] == 0.2, sigma [1][1] == 0.1, sigma [2][2] == 0.3, sigma [3][3] == 0.1,
sigma [0][1] >= 0, sigma [0][2] >= 0, sigma [1][0] >= 0, sigma [2][0] >= 0, sigma [2][3] >= 0, sigma [3][2] >= 0,
sigma [1][2] <= 0, sigma [1][3] <= 0, sigma [2][1] <= 0, sigma [3][1] <= 0,
sigma >> 0
]

problem = cp.Problem (objective, constraints) 

problem.solve ()

print (f'status: {problem.status} ')

print(f'optimal value: {problem.value} ')

print (f'optimal var: In {sigma.value}')

diagonal = np.diag([0.2, 0.1, 0.3, 0.1])

objective_diagonal = x.T @ diagonal @ x

print(f'optimal diagonal: {objective_diagonal[0,0]}')