import os
import numpy as np
from matplotlib import pyplot as plt
from model import Model
from models.accel.gen_accel_mesh import AccelBlankMeshGenerator, base_dir
from util import plot_mesh

d = 400
g = AccelBlankMeshGenerator(5200, 2400, 320, 69, d)
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

fs = []
n_els = []

for i in range(100):

    Us, f, w_sq = model.solve_modes(1)
    print(f.squeeze())
    fs.append(f.squeeze())

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
    print(n_el, "/", len(g.domain))
    n_els.append(n_el)

    idx_active = np.argsort(a_tilde)
    idx_active = [i for i in idx_active[x[idx_active] == 1] if i in g.domain]

    # idx_inactive = np.argsort(a_tilde)
    # idx_inactive = [i for i in idx_inactive[x[idx_inactive] < 1] if i in g.domain]

    if n_el > V_goal * len(g.domain):

        x[idx_active[:min(int(np.ceil(n_el * ER)), len(idx_active))]] = x_min

        # for i in range(min(int(np.ceil(n_el * AR_max)), len(idx_inactive))):
        #     if a_tilde[idx_inactive[i]] > 0:
        #         x[idx_inactive[i]] = 1

    else:

        break

    model.K = sum([((x_min - x_min ** p) / (1 - x_min ** p) * (1 - x_i ** p) + x_i ** p) * K_i for K_i, x_i in zip(model.K_els, x)])
    model.M = sum([x_i * M_i for M_i, x_i in zip(model.M_els, x)])

filename = f"accel_optimized_{d}um"

dir = os.path.dirname(os.path.join(base_dir, filename, f'{filename}.txt'))
if not os.path.exists(dir):
    os.makedirs(dir)

np.savetxt(os.path.join(base_dir, filename, f"{filename}_iterations.txt"), np.c_[fs, n_els, np.full_like(n_els, len(g.domain))], fmt="%f, %i, %i")
fig, ax = plt.subplots()
ln1 = ax.plot(fs, label='first eigenfrequency')
ax.set_ylabel("f (Hz)")
ax2 = ax.twinx()
ln2 = ax2.plot(np.array(n_els)/len(g.domain), label='volume fraction', color='orange')
ax.set_xlabel("iterations")
lines = ln1+ln2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels)
fig.savefig(os.path.join(base_dir, filename, f"{filename}_iterations.pdf"))

base = plot_mesh(nodes, elems, title=f"starting mesh ({d}um element)")

g.remove_elems(np.where(x < 1))

g.dump_inp(E=E, nu=nu, rho=rho, filename=f"{filename}/{filename}")
g.dump_txt(filename=f"{filename}/{filename}")

nodes = np.c_[g.x, g.y, g.z]
elems = g.elems
fixed = g.fixed

model = Model(nodes, elems, fixed, E=E, nu=nu, rho=rho)
model.calc_stiffness()
model.calc_mass()
Us, f, w_sq = model.solve_modes(1)
print(f.squeeze())
np.savetxt(os.path.join(base_dir, filename, f"{filename}_displacement.txt"), np.c_[np.arange(len(g.nodes))+1, Us[0]], fmt="%i %e %e %e")

stationary = plot_mesh(nodes, elems, title=f"optimized design ({d}um element)")
displaced = plot_mesh(nodes, elems, Us[0], 0.1, title=f"first eigenmode ({d}um element)")

base.savefig(os.path.join(base_dir, filename, f"{filename}_base.pdf"))
stationary.savefig(os.path.join(base_dir, filename, f"{filename}_stationary.pdf"))
displaced.savefig(os.path.join(base_dir, filename, f"{filename}_displaced.pdf"))

plt.show()