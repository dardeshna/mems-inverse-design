import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from model import Model
import scipy.sparse

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
mode_ref =  np.genfromtxt('models/beam20p/ref_mode.txt', delimiter="")[:, 1:]

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

U, F = model.solve_displacement(loaded, np.full(loaded.shape, 1), np.ones(loaded.shape))

print(np.max(np.abs(U-U_ref)))
print(np.max(np.abs(F-F_ref)))

Us, f, w_sq = model.solve_modes(10)

print(f)
print(np.max(np.abs(Us[1]-mode_ref)))