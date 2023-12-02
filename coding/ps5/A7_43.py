import numpy as np
import cvxpy as cp 
from data.consistent_decile_data import *

############  B  ###############

v = cp.Variable((9, 1))
theta = cp.Variable((9, n))

y = y.reshape(1, 500)

var = v + theta @ X.T - y

const = np.linspace(-0.4,0.4, 9)
const = np.flip(const)

const = const.reshape(9, 1)

obj = cp.Minimize(0.5 * cp.sum(cp.norm1(var)) + cp.sum(cp.multiply(const,var)))

constraints = [cp.sum(cp.norm1(theta[i + 1] + theta[i])) <= v[i + 1] - v[i] for i in range(8)] 

problem = cp.Problem(obj, constraints)

answer = problem.solve()

v_hat = v.value
theta_hat = theta.value
q_hat = v_hat + theta_hat @ X.T

print('Percentile')
for j in range(9):
    fraction = np.mean(np.where(y <= q_hat[j, :], np.ones_like(y), np.zeros_like(y)))
    print(f'{0.10*j:.1f}: {fraction:.3f}')
    
print(f'The optimal values for v are: {v_hat.reshape(1,-1)}')
print(f'The optimal values for theta are: {theta_hat[0,:]}')


############  C  ###############

v = cp.Variable((9, 1))
theta = cp.Variable((9, n))

y = y.reshape(1, 500)

var = v + theta @ X.T - y

const = np.linspace(-0.4,0.4, 9)
const = np.flip(const)

const = const.reshape(9, 1)

obj = cp.Minimize(0.5 * cp.sum(cp.norm1(var)) + cp.sum(cp.multiply(const,var)))

constraints = [] 

problem = cp.Problem(obj, constraints)

answer = problem.solve()

v_hat = v.value
theta_hat = theta.value

q_hat = v_hat + theta_hat @ X.T
q_hat = q_hat.reshape(500, 9)


print(f'The optimal values for v (without constraints) are: {v_hat.reshape(1,-1)}')
print(f'The optimal values for theta (without constraints) are: {theta_hat[0,:]}')

I = np.eye(9)
D = np.diff(I, axis=0)


uno = np.ones(5000)
meno_uno = np.ones(5000) * -1
sample_from = np.hstack((uno, meno_uno))

x_inc = np.ones((10,1))

res = D @ (v_hat + theta_hat @ x_inc)
first_negative = np.argmax(res < 0)

q_ts = (v_hat + theta_hat @ x_inc)[first_negative:first_negative + 2]

print(f'Setting x_inc to be the vector of ones: q{first_negative}: {q_ts[0]}, q{first_negative + 1}: {q_ts[1]} the difference q{first_negative + 1} - q{first_negative} is: {res[first_negative]}')