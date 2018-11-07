import numpy as np

A = np.array([[-1, 2, -9, 4]])
A[A < 0] = 0
print(A)
