import numpy as np
import scipy
import scipy.integrate
import scipy.linalg
import scipy.sparse

nodes = np.genfromtxt('../models/beam20p/nodes.txt', delimiter=",")[:, 1:]
elems =  np.genfromtxt('../models/beam20p/elems.txt', delimiter=",", dtype=int)[:, 1:]
fixed =  np.genfromtxt('../models/beam20p/fixed.txt', delimiter=",", dtype=int)
loaded =  np.genfromtxt('../models/beam20p/loaded.txt', delimiter=",", dtype=int)
u_ref =  np.genfromtxt('../models/beam20p/ref_disp.txt', delimiter="")[:, 1:]
f_ref =  np.genfromtxt('../models/beam20p/ref_force.txt', delimiter="")[:, 1:]

print("fixed: ", fixed)
print("loaded: ", loaded)

num_n = nodes.shape[0]
num_free = nodes.shape[0] - fixed.shape[0]
num_el = elems.shape[0]

map = dict(zip([x for x in np.arange(num_n)+1 if x not in fixed], range(num_free)))
get_coord = lambda node, dof : 3 * map[node] + dof - 1 # node and dof are 1 indexed

K_ref = np.genfromtxt('../models/beam20p/get_stiffness/beam20p.sti')
K_ref = scipy.sparse.coo_matrix((K_ref[:,2], ((K_ref[:,0]-1).astype(dtype=int), (K_ref[:,1]-1).astype(dtype=int))), shape=(3*num_free, 3*num_free)).toarray()
K_ref = K_ref.T + K_ref - np.diag(np.diag(K_ref))

M_ref = np.genfromtxt('../models/beam20p/get_stiffness/beam20p.mas')
M_ref = scipy.sparse.coo_matrix((M_ref[:,2], ((M_ref[:,0]-1).astype(dtype=int), (M_ref[:,1]-1).astype(dtype=int))), shape=(3*num_free, 3*num_free)).toarray()
M_ref = M_ref.T + M_ref - np.diag(np.diag(M_ref))

w, v = scipy.linalg.eig(K_ref, M_ref)
print(np.sort(np.sqrt(w) / (2*np.pi))[:10])

u_ref = [u_ref[x] for x in np.arange(num_n) if x+1 not in fixed]
u_ref = np.array(u_ref).flatten()

f_ref = [f_ref[x] for x in np.arange(num_n) if x+1 not in fixed]
f_ref = np.array(f_ref).flatten()

f = K_ref @ u_ref.flatten()
f_ref = f_ref.flatten()

print([f_ref[get_coord(x, 2)] for x in loaded])
print([f[get_coord(x, 2)] for x in loaded])