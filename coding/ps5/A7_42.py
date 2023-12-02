import numpy as np 
import cvxpy as cp 

M = 101

uniform = np.linspace(0, 1, M).reshape(101,1)


W = cp.Variable((M,M))

uniform = np.linspace(0., 1., M)

obj = W[50,50]

constraints = (
    [W[0, i] == 0 for i in range(M)] +
    [W[i, 0] == 0 for i in range(M)] + 
    [W[i, M-1] == uniform[i] for i in range(M)] +
    [W[M-1, i] == uniform[i] for i in range(M)] +
    [W[i + 1, j + 1] - W[i, j + 1] - W[i + 1, j] + W[i , j] >= 0 for j in range(M - 1) for i in range(M - 1)] +
    [W[i, j] <= 1 for j in range(M) for i in range(M)] +
    [W[i, j] >= 0 for j in range(M) for i in range(M)] +
    [W[50, 90] == 0.45,
    W[10, 60] == 0.06,
    W[30, 30] == 0.09,
    W[60, 20] == 0.12]
)




min_problem = cp.Problem(cp.Minimize(obj), constraints)
min_problem.solve()

minimum_objective = W.value[50,50]


max_problem = cp.Problem(cp.Maximize(obj), constraints)
max_problem.solve()
maximum_objective = W.value[50,50]

print(f'The minimum value obtained for W_m,m is: {minimum_objective}\n, instead the maximum value obtained is: {maximum_objective}')





