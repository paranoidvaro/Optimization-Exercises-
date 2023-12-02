import numpy as np
import cvxpy as cp 

np.random.seed(10)

(m, n) = (30, 10)

A = np.random.rand(m, n)
b = np.random.rand(m, 1)
c_nom = np.ones((n, 1)) + np.random.rand(n, 1)

x = cp.Variable((n,1))

# Create separate λ variables for each type of constraint
λ_upper = cp.Variable((n,1))
λ_lower = cp.Variable((n,1))
λ_upper_mean = cp.Variable((1,1))
λ_lower_mean = cp.Variable((1,1))

g_upper = 1.25 * c_nom
g_lower = -0.75 * c_nom
g_upper_mean = 1.1 * np.mean(c_nom) 
g_lower_mean = -0.9 * np.mean(c_nom)

# The objective function is now a sum of four terms
objective = cp.Minimize((λ_upper.T @ g_upper) + (λ_lower.T @ g_lower) + λ_upper_mean * g_upper_mean + λ_lower_mean * g_lower_mean)

# The constraints are now represented individually
constraints = [
    A @ x >= b,
    λ_upper >= 0,
    λ_lower >= 0,
    λ_upper_mean >= 0,
    λ_lower_mean >= 0,
    λ_upper - λ_lower + 0.1 * λ_upper_mean - 0.1 * λ_lower_mean == x,
]

problem = cp.Problem(objective, constraints)
problem.solve()

x_robust = x.value

constraint_2 = [
    A @ x >= b
]

obj_2 = cp.Minimize(c_nom.T @ x)
prob_2 = cp.Problem(obj_2, constraint_2)
prob_2.solve()

x_nom = x.value

print(f'nominal costs: {(c_nom.T @ x_robust)[0][0]} (for optimal x for robust)) {(c_nom.T @ x_nom)[0][0]} (for optimal x for nominal)')


c = cp.Variable((n,1))


F = np.vstack((np.eye(n), -np.eye(n), np.ones((1,n))/n, -np.ones((1,n))/n))
g = np.vstack((1.25*c_nom, -0.75*c_nom, 1.1*0.1*np.sum(c_nom), -0.9*0.1*np.sum(c_nom)))


constraints_wc = [
    F @ c <= g
]


obj_wc_nom = cp.Maximize(c.T @ x_nom)
obj_wc_robust = cp.Maximize(c.T @ x_robust)

prob_wc_nom = cp.Problem(obj_wc_nom, constraints_wc)
prob_wc_robust = cp.Problem(obj_wc_robust, constraints_wc)


prob_wc_nom.solve()
prob_wc_robust.solve()

print(f'worst case costs: {prob_wc_robust.value} (for optimal x for robust)) {prob_wc_nom.value} (for optimal x for nominal)')
