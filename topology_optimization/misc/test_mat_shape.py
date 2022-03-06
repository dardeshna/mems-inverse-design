import numpy as np
import scipy.linalg

# A = np.arange(3*20).reshape((3, 20))
# Q = scipy.linalg.block_diag(*np.broadcast_to(A, (3,*A.shape)))
# print(A)

# print(Q.reshape(-1, 3, 20).transpose().reshape(-1,9).transpose())

A = np.arange(20*20).reshape((20, 20))
Q = scipy.linalg.block_diag(*np.broadcast_to(A, (3,*A.shape)))