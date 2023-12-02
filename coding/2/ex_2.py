from data.max_util_wireless_data import *
import numpy as np
import cvxpy as cp

### B)

# f = cp.Variable((n,1))


# b_reshaped = b.reshape((8, 1))

# c_reshaped = c.reshape((20, 1))

# obj = cp.Minimize(-cp.sum([cp.sqrt(i) for i in f]))

# constraints = [A @ R @ f <= b_reshaped, R @ f <= c_reshaped, f >= 0]

# problem = cp.Problem(obj, constraints)
# problem.solve()

# print(f'optimal value: {problem.value}')
# print(f'status: {problem.status}')
# print(f'optimal var: {f.value}')



# full_capacity_links = np.argwhere(np.isclose(R @ f.value - c_reshaped, 0, atol=0.001))

# print(f'the links operating at full capacity are i = {[link for link in full_capacity_links[:,0]]}')

#### C)

f = cp.Variable((n,1))


b_reshaped = b.reshape((8, 1))

c_reshaped = c.reshape((20, 1))

obj = cp.Minimize(-cp.sum([cp.sqrt(i) for i in f]))

constraints = [R @ f <= c_reshaped, f >= 0]

problem = cp.Problem(obj, constraints)
problem.solve()

print(f'optimal value: {problem.value}')
print(f'status: {problem.status}')
print(f'optimal var: {f.value}')



full_capacity_links = np.argwhere(np.isclose(R @ f.value - c_reshaped, 0, atol=0.001))

print(f'the links operating at full capacity are i = {[link for link in full_capacity_links[:,0]]}')

