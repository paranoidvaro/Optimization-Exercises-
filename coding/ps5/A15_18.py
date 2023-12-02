import numpy as np
import cvxpy as cp
from data.neural_signal_data import *

a = cp.Variable(N)

lam = 2


obj = cp.Minimize(cp.sum(cp.square(y - cp.conv(s,a).flatten())) / T +  lam * cp.norm1(a)) 

constraints = [a >= 0]

problem = cp.Problem(obj, constraints)

problem.solve()

a_hat = a.value

non_zero = find_nonzero_entries(a_hat)

non_zero_true = find_nonzero_entries(a_true)

print('B)')
print(f'True:      {non_zero_true}')
print(f'Estimated: {non_zero}')

visualize_estimate(a_hat)
plt.show()

tau = np.where(a_hat <= 0.01)[0]
# print(tau)
minimum = np.max(tau)
minimum = np.min(tau)

tau_0 = np.where(a_hat >= 0.01)[0]

tau_0_merging_couples = np.array([tau_0[0]] + [round(np.mean(tau_0[1:3]))] + list(tau_0[3::]))
# print(tau_0_merging_couples)

final_tau = [i for i in np.linspace(0,N-1,N).astype(int) if i not in tau_0_merging_couples]


################ C.a ##################

a_polished = cp.Variable(N)


obj = cp.Minimize(cp.sum(cp.square(y - cp.conv(s,a_polished).flatten())) / T) 

constraints = [a_polished >= 0, a_polished[final_tau] == 0]

problem = cp.Problem(obj, constraints)

problem.solve()

a_hat_poli = a_polished.value

non_zero = find_nonzero_entries(a_hat_poli)

non_zero_true = find_nonzero_entries(a_true)

print('C (polished + merged)')
print(f'True:      {non_zero_true}')
print(f'Estimated: {non_zero}')

visualize_polished(a_hat_poli)
plt.show()

