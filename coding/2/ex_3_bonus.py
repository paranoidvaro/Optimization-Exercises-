import numpy as np
import cvxpy as cp 
import matplotlib.pyplot as plt


assets = 500

scenarios = 20



probabilities = np.random.randn(scenarios)


standard_probabilities = (probabilities - np.min(probabilities)) / (np.max(probabilities) - np.min(probabilities))

normal_probabilities = standard_probabilities / np.sum(standard_probabilities)

normal_probabilities = normal_probabilities.reshape(20,1)

matrix = np.random.standard_cauchy((assets, scenarios))


plt.scatter(matrix[:,0], [i for i in range(len(matrix[:,0]))])


x = cp.Variable((assets, 1))



# print(normal_probabilities.shape)

# print(matrix.shape)

# print(x.shape)

# print(matrix.T @ x)

obj = cp.Maximize(cp.sum(normal_probabilities.T @ cp.log(matrix.T @ x)))

problem = cp.Problem(obj)

problem.solve()

id = np.eye(assets,assets)


constraints = [x >= 0, cp.sum(id.T @ x) == 1]

# print(matrix)
print(x.value)

# plt.show()




