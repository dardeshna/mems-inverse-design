import os
import numpy as np
from matplotlib import pyplot as plt
from model import Model
from models.accel.gen_accel_mesh import AccelBlankMeshGenerator, base_dir
from util import plot_mesh


g = AccelBlankMeshGenerator(5200, 2400, 320, 69, 400)
nodes = np.c_[g.x, g.y, g.z]
elems = g.elems
fixed = g.fixed

E = 170000
nu = 0.280
rho = 2.329e-09

model = Model(nodes, elems, fixed, E=E, nu=nu, rho=rho)
model.calc_stiffness()
model.calc_mass()

x = np.ones(model.num_el)

x_min = 1e-6
p = 3
r_min = 0.8
V_goal = 0.5
ER = 0.02
AR_max = 0.02

r = np.sqrt((g.x - g.cent_x[...,None]) ** 2 + (g.y - g.cent_y[...,None]) ** 2 + (g.z - g.cent_z[...,None]) ** 2)

weight = np.clip(r_min - r, 0, r_min)
weight /= np.sum(weight, axis=1, keepdims=True)

a_tilde = None

for i in range(100):

    Us, f, w_sq = model.solve_modes(1)

    print(f)

    U = Us[0][list(model.map.keys())].flatten()
    w = np.lib.scimath.sqrt(w_sq[0])

    a = [1/(2*w) * U.T @ ((1-x_min) / (1-x_min**p) * x_i ** (p-1) * K_i - w ** 2 / p * M_i) @ U for K_i, M_i, x_i in zip(model.K_els, model.M_els, x)]

    a_nodal = [np.mean([s for s,e in zip(a, model.elements) if i in e]) for i in range(model.num_nodes)]
        
    a_hat = weight @ a_nodal

    if i > 0:
        a_tilde = 1/2 * (a_tilde + a_hat)
    else:
        a_tilde = a_hat

    n_el = np.sum([1 for i in range(model.num_el) if i in g.domain and x[i] == 1])

    idx_active = np.argsort(a_tilde)
    idx_active = [i for i in idx_active[x[idx_active] == 1] if i in g.domain]

    idx_inactive = np.argsort(a_tilde)
    idx_inactive = [i for i in idx_inactive[x[idx_inactive] < 1] if i in g.domain]

    if n_el > V_goal * len(g.domain):

        x[idx_active[:min(int(np.ceil(n_el * ER)), len(idx_active))]] = x_min

        # for i in range(min(int(np.ceil(n_el * AR_max)), len(idx_inactive))):
        #     if a_tilde[idx_inactive[i]] > 0:
        #         x[idx_inactive[i]] = 1

    else:

        break

    model.K = sum([((x_min - x_min ** p) / (1 - x_min ** p) * (1 - x_i ** p) + x_i ** p) * K_i for K_i, x_i in zip(model.K_els, x)])
    model.M = sum([x_i * M_i for M_i, x_i in zip(model.M_els, x)])

    print(n_el, "/", len(g.domain))

g.remove_elems(np.where(x < 1))

filename = "accel_optimized_400um"
g.dump_inp(E=E, nu=nu, rho=rho, filename=f"{filename}/{filename}")
g.dump_txt(filename=f"{filename}/{filename}")

nodes = np.c_[g.x, g.y, g.z]
elems = g.elems
fixed = g.fixed

model = Model(nodes, elems, fixed, E=E, nu=nu, rho=rho)
model.calc_stiffness()
model.calc_mass()
Us, f, w_sq = model.solve_modes(1)
print(f)

stationary = plot_mesh(nodes, elems)
displaced = plot_mesh(nodes, elems, Us[0], 0.1)

stationary.savefig(os.path.join(base_dir, filename, f"{filename}_stationary.png"))
displaced.savefig(os.path.join(base_dir, filename, f"{filename}_displaced.png"))

plt.show()