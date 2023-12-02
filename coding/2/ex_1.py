#### Problem Data Generation ####
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt



np.random.seed(0)

(m, n) = (300, 100)
A = np.random.rand(m, n)
A = np.asmatrix(A)

b = A.dot(np.ones((n, 1))) / 2
b = np.asmatrix(b)

c = -np.random.rand(n, 1)
c = np.asmatrix(c)

# print(A.shape)
# print(b.shape)
# print(c.shape)

# we start with the relaxation

x = cp.Variable((n,1))
obj = cp.Minimize(c.T @ x)

constraints = [A @ x <= b, x >= 0, x <= 1]

problem = cp.Problem(obj, constraints)
problem.solve()

print(f'optimal value: {problem.value}')
print(f'status: {problem.status}')
print(f'optimal var: {x.value}')
# print(x.T.shape)
lower_bound = c.T @ x.value
# print(lower_bound.shape)

vector_t = np.linspace(0,1,100)
matrix_t= np.repeat(vector_t[:, np.newaxis], A.shape[1], axis=1)

matrix_x = np.vstack([x.T.value]*n)

final = matrix_x >= matrix_t
final_int = final.astype(int)
objective_value_vector = c.T @ final_int.T
# print(objective_value_vector.shape)
# print(A.shape)
# print(b.shape)
maximum_constraint_violation = np.max(A @ final_int.T - b, axis = 0)
transposed_obj = objective_value_vector.T
ones = np.ones((n, 1))
lower_bound_vec = np.multiply(ones, lower_bound)
# print(lower_bound_vec.shape)
t_index = np.argmax(transposed_obj >= lower_bound_vec)
zeros = np.zeros((n,1))

# # print(c.shape)
# First plot
plt.figure()
plt.plot(vector_t, transposed_obj)
plt.plot(vector_t, lower_bound_vec)
plt.xlabel('threshold t')
plt.axvline(vector_t[t_index], color='green')

# Add a text label for the line at x = 2
plt.text(vector_t[t_index], max(transposed_obj)/2, 't = {}'.format(round(vector_t[t_index],2)))

# Add shaded regions
plt.fill_between(vector_t, np.min(transposed_obj), np.max(transposed_obj), where=vector_t <= vector_t[t_index], color='red', alpha=0.3)
plt.fill_between(vector_t, np.min(transposed_obj), np.max(transposed_obj), where=vector_t > vector_t[t_index], color='lightblue', alpha=0.3)

plt.ylabel('objective')
plt.title('Plot of Objective')
plt.show()

# Second plot
plt.figure()
plt.plot(vector_t, maximum_constraint_violation.T)
plt.plot(vector_t, zeros)
plt.xlabel('threshold t')

# Add shaded regions
plt.fill_between(vector_t, np.min(maximum_constraint_violation), np.max(maximum_constraint_violation), where=vector_t > vector_t[t_index], color='red', alpha=0.3)
plt.fill_between(vector_t, np.min(maximum_constraint_violation), np.max(maximum_constraint_violation), where=vector_t <= vector_t[t_index], color='lightblue', alpha=0.3)

plt.ylabel('maximum constraint violation')
plt.title('Plot of Maximum Constraint Violation')
plt.show()


# print(maximum_constraint_violation.shape)

L = problem.value

U = transposed_obj[t_index]

print(f'L is: {L}')
print(f'U is: {U[0,0]}')

print(f'U - L = {(U - L)[0,0]}')








