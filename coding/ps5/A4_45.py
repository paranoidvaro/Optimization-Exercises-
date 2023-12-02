import cvxpy as cp 
import numpy as np 

alfa_1 = 0.2
alfa_2 = 0.1
alfa_3 = 0.3
alfa_4 = 1

c_max = 60

T_min = 10
T_max = 40

r_min = 35
r_max = 80

w_min = 3
w_max = 4

T = cp.Variable(pos=True)
r = cp.Variable(pos=True)
w = cp.Variable(pos=True)

obj = alfa_4 * T * cp.square(r)

constraints = [
    alfa_1 * T * r * cp.inv_pos(w) + alfa_2 * r + alfa_3 * w <= c_max,
    T_min <= T,
    T <= T_max,
    r_min <= r,
    r <= r_max,
    w_min <= w,
    w <= w_max,
    w <= 0.1 * r]

problem = cp.Problem(cp.Maximize(obj), constraints)
problem.solve(gp=True)


print(f'Problem Status: {problem.status}')
print(f'Maximum Heat Flow: {problem.value}\n')
print(f'Reached with the following configuration:\n     T = {T.value}\n     r = {r.value}\n     w = {w.value}')