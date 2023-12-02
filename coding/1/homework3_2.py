# data for vehicle speed scheduling problem.
# contains quantities: n, a, b, c, d, smin, smax, tau_min, tau_max
import cvxpy as cp
import matplotlib.pyplot as plt
from data.veh_speed_sched_data import * 

v = cp.Variable(n)

constraints = [(d / smax) <= v,  (d / smin) >= v, cp.cumsum(v) >= tau_min, cp.cumsum(v) <= tau_max]

objective = cp.Minimize(cp.sum(a * cp.multiply(cp.square(d), cp.inv_pos(v)) + b * d + c * v))

problem = cp.Problem(objective, constraints) 

result = problem.solve()

print(v.value, result)

index_vector = np.array([i for i in range(100)])
# print(d / np.array(v.value))
plt.plot(index_vector, d / np.array(v.value))
plt.show()