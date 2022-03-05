import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import time

from c3d20 import elem_K, elem_M

class Model():
    """Represents a finite element model with c3d20 brick elements. All values are zero indexed!
    """

    def __init__(self, nodes, elements, fixed, **kwargs):

        self.nodes = np.array(nodes)
        self.elements = np.array(elements)
        self.fixed = np.array(fixed)

        self.num_nodes = self.nodes.shape[0]
        self.num_el = self.elements.shape[0]
        self.num_fixed = self.fixed.shape[0]
        self.num_free = self.num_nodes - self.num_fixed

        self.map = dict(zip([x for x in range(self.num_nodes) if x not in fixed] + np.sort(fixed).tolist(), range(self.num_nodes)))
        self.imap = {v:k for k,v in sorted(self.map.items(), key = lambda x: x[0])}

        kwargs = {'E': 1, 'nu': 0, 'rho': 1, **kwargs}

        for k in ('E', 'nu', 'rho'):
            if hasattr(kwargs[k], '__len__') and len(kwargs[k]) == self.num_el:
                setattr(self, k, np.array(kwargs[k]))
            else:
                setattr(self, k, np.full((self.num_el,), kwargs[k]))
    
    def idx(self, node, dof):

        return 3 * self.map[node] + dof

    def build_mat(self, elem_mat, *args):

        row = []
        col = []
        data = []
        for e_i, e in enumerate(self.elements):
            n_coords = self.nodes[e]
            mat_elem = elem_mat(n_coords, *[i[e_i] for i in args])
            for i, n_i in enumerate(e):
                for j, n_j in enumerate(e):
                    for k in range(3):
                        for l in range(3):
                            row.append(self.idx(n_i, k))
                            col.append(self.idx(n_j, l))
                            data.append(mat_elem[3*i+k,3*j+l])

        return scipy.sparse.coo_matrix((data, (row, col)), shape=(3*self.num_nodes, 3*self.num_nodes)).asformat("csr")

    def calc_stiffness(self):
        self.K = self.build_mat(elem_K, self.E, self.nu)
        return self.K

    def calc_mass(self):
        self.M = self.build_mat(elem_M, self.rho)
        return self.M

    def solve_displacement(self, nodes, dofs, loads):

        K_reduced = self.K[:self.num_free*3, :self.num_free*3]
        F_reduced = np.zeros(3 * self.num_free)

        for n, d, f in zip(nodes, dofs, loads):
            F_reduced[self.idx(n, d)] = f

        U_reduced = scipy.sparse.linalg.spsolve(K_reduced, F_reduced)
        
        U = np.append(U_reduced, np.zeros(3 * self.num_fixed))
        F = self.K @ U

        idx = np.array(list(self.imap.keys()))

        return U.reshape(-1,3)[idx], F.reshape(-1,3)[idx]

# m = Model(np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), E=2, rho=1)
# print(m.rho)

# nodes = np.genfromtxt('models/single_elem/nodes.txt', delimiter=",")[:, 1:]
# elems = [np.arange(20),]
# fixed = []

nodes = np.genfromtxt('models/beam20p/nodes.txt', delimiter=",")[:, 1:]
elems =  np.genfromtxt('models/beam20p/elems.txt', delimiter=",", dtype=int)[:, 1:]-1
fixed =  np.genfromtxt('models/beam20p/fixed.txt', delimiter=",", dtype=int)-1
loaded =  np.genfromtxt('models/beam20p/loaded.txt', delimiter=",", dtype=int)-1
U_ref =  np.genfromtxt('models/beam20p/ref_disp.txt', delimiter="")[:, 1:]
F_ref =  np.genfromtxt('models/beam20p/ref_force.txt', delimiter="")[:, 1:]

model = Model(nodes, elems, fixed, E=210000, nu=0.3, rho=7.8E-9)
K = model.calc_stiffness()[:model.num_free*3, :model.num_free*3]
M = model.calc_mass()[:model.num_free*3, :model.num_free*3]

# K_ref = np.genfromtxt('single_elem/single_elem.sti')
K_ref = np.genfromtxt('models/beam20p/get_stiffness/beam20p.sti')
K_ref = scipy.sparse.coo_matrix((K_ref[:,2], (K_ref[:,0]-1, K_ref[:,1]-1)), shape=(3*model.num_free, 3*model.num_free)).toarray()
K_ref = K_ref.T + K_ref - np.diag(np.diag(K_ref))

print(np.max(np.abs(K-K_ref)))

# M_ref = np.genfromtxt('single_elem/single_elem.mas')
M_ref = np.genfromtxt('models/beam20p/get_stiffness/beam20p.mas')
M_ref = scipy.sparse.coo_matrix((M_ref[:,2], (M_ref[:,0]-1, M_ref[:,1]-1)), shape=(3*model.num_free, 3*model.num_free)).toarray()
M_ref = M_ref.T + M_ref - np.diag(np.diag(M_ref))

print(np.max(np.abs(M-M_ref)))

# w, v = np.linalg.eig(M.toarray())
# print(w[-10:])

start = time.time()
w, v = scipy.linalg.eigh(K.toarray(), M.toarray())
print("eigh: ", time.time()-start)

# print(np.sort(np.abs(w))[:10])
print(np.sort(np.sqrt(w) / (2*np.pi))[:10])

start = time.time()
w, v = scipy.sparse.linalg.eigsh(K, 10, M, which='LM', sigma=1)
print("eigsh: ", time.time()-start)

# print(np.sort(np.abs(w))[:10])
print(np.sort(np.sqrt(w) / (2*np.pi)))

U, F = model.solve_displacement(loaded, np.full(loaded.shape, 1), np.ones(loaded.shape))

print(np.max(np.abs(U-U_ref)))
print(np.max(np.abs(F-F_ref)))