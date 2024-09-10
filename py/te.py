import numpy as np

D = np.diag([34, 45, 27])
L = np.array([[0, 0, 0], [8, 0, 0], [1, 9, 0]])
U = np.array([[0, 6, 5], [0, 0, 5], [0, 0, 0]])

S2 = np.matmul(-np.linalg.inv(D + L), U)
S1 = np.matmul(-np.linalg.inv(D + U), L)
S = np.matmul(S1, S2)

a = np.abs(np.linalg.eigvals(S)[2])

m0 = 1
m1 = 1 / a
m2 = 2 / a * m1 - m0
m3 = 2 / a * m2 - m1
print(m2, m3)
