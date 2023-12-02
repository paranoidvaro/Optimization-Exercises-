import cvxpy as cp
import numpy as np 

X = cp.Variable (16)

constraints = [
        X >= 0, 
        X <= 1,
        cp.sum(X) == 1,
        X[8] + X[9] + X[10] + X[11] + X[12] + X[13] + X[14] + X[15] == 0.9,
        X[4] + X[5] + X[6] + X[7] + X[12]+X[13]+X[14]+X[15] == 0.9,
        X[2]+X[3]+X[6]+X[7]+X[10]+X[11]+X[14]+X[15]== 0.1,
        X[14]+X[10] == 0.07,
        6*X[4]-4*X[5]+6*X[12]-4*X[13]== 0
]
object1 = cp.Minimize(X[1]+X[3]+X[5]+X[7]+X[9]+X[11]+X[13]+X[15]) #0.48
object2 = cp.Minimize(-(X[1]+X[3]+X[5]+X[7]+X[9]+X[11]+X[13]+X[15])) #0.6

problem = cp.Problem(object2, constraints)

problem.solve()

print(f'status:{problem.status}')
print(f'optimal value: {problem.value}')
print(f'optimal var: {X.value}')

