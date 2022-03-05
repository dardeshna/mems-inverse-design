
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import sys


K_ref = np.genfromtxt('c3d4.sti')
K_ref = scipy.sparse.coo_matrix((K_ref[:,2], ((K_ref[:,0]-1).astype(dtype=int), (K_ref[:,1]-1).astype(dtype=int))), shape=(12, 12)).toarray()

M_ref = np.genfromtxt('c3d4.mas')
M_ref = scipy.sparse.coo_matrix((M_ref[:,2], ((M_ref[:,0]-1).astype(dtype=int), (M_ref[:,1]-1).astype(dtype=int))), shape=(12, 12)).toarray()

np.set_printoptions(threshold=sys.maxsize,precision=2)

print(np.round(K_ref))

K_ref = K_ref + K_ref.T - np.diag(np.diag(K_ref))

print(np.round(K_ref))

# w, v = scipy.linalg.eig(K_ref, M_ref)
# w, v = scipy.sparse.linalg.eigs(K_ref,10,M_ref,which='LM')
# print(np.sort(np.sqrt(w) / (2*np.pi)))