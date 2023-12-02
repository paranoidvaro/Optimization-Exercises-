import numpy as np 
import cvxpy as cp

from data.buy_hold_sell_data import *


W = cp.Variable(n)

obj = cp.Maximize(cp.sum(W[0 : nb]) - cp.norm1(W[nb : nb + nh]) - cp.sum(W[nb + nh ::]))

identity = np.eye(n)

constraints = [W[0 : nb] >= 0, W[nb + nh ::] <= 0, cp.sum(W) == 1, cp.norm1(W) <= L, W.T @ Sigma @ W <= cp.power(sigma, 2)]

problem = cp.Problem(obj, constraints)

result = problem.solve()

W_hat = W.value

w_b_hat = W_hat[0 : nb]
w_h_hat = W_hat[nb : nb + nh]
w_s_hat = W_hat[nb + nh ::]

print(problem.status)

print(f'The resulting weights for:\nw_buy are: {[round(i,2) for i in w_b_hat]},\nw_hold are: {[round(i,2) for i in w_h_hat]},\nw_sell are: {[round(i,2) for i in w_s_hat]}')
print(f'With a worst case return of: {result:.2f}')

################# worst case mu ###################

mus = np.sign(W_hat)
print('mu',mus)

############## C ###############

W = cp.Variable(n)

constraints = [cp.sum(W) == 1, cp.sum(cp.norm1(W)) <= L, W.T @ Sigma @ W <= cp.square(sigma)]

objective = cp.Maximize(cp.sum(W[0: nb]) - cp.sum(W[nb + nh::]))

problem = cp.Problem(objective, constraints)

result = problem.solve()

W_naive_hat = W.value

w_b_hat = W_naive_hat[0 : nb]
w_h_hat = W_naive_hat[nb : nb + nh]
w_s_hat = W_naive_hat[nb + nh ::]


print(f'The resulting weights (naive) for:\nw_buy are: {[round(i,2) for i in w_b_hat]},\nw_hold are: {[round(i,2) for i in w_h_hat]},\nw_sell are: {[round(i,2) for i in w_s_hat]}')
print(f'With a worst case return of: {result:.2f}')

mus = np.sign(W_naive_hat)
print('mu',mus)





