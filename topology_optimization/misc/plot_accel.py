import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from matplotlib import pyplot as plt
from util import plot_mesh
from models.accel.gen_accel_mesh import AccelBlankMeshGenerator

for d,sgn in ((400,1), (100,-1)):

    filename = f"accel_optimized_{d}um"

    g = AccelBlankMeshGenerator(5200, 2400, 320, 69, d)
    nodes = np.c_[g.x, g.y, g.z]
    elems = g.elems

    fs, n_els, _ = np.genfromtxt(f'../models/accel/{filename}/{filename}_iterations.txt', delimiter=",").T
    fig, ax = plt.subplots()
    ln1 = ax.plot(fs, label='first eigenfrequency')
    ax.set_ylabel("f (Hz)")
    ax2 = ax.twinx()
    ln2 = ax2.plot(np.array(n_els)/len(g.domain), label='volume fraction', color='orange')
    ax.set_xlabel("iterations")
    lines = ln1+ln2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels)
    fig.savefig(os.path.join(f'../models/accel/{filename}/{filename}_iterations.pdf'))

    base = plot_mesh(nodes, elems, title=f"starting mesh ({d}um element)")

    nodes = np.genfromtxt(f'../models/accel/{filename}/{filename}_nodes.txt', delimiter=",")[:, 1:]
    elems =  np.genfromtxt(f'../models/accel/{filename}/{filename}_elems.txt', delimiter=",", dtype=int)[:, 1:]-1
    disp =  np.genfromtxt(f'../models/accel/{filename}/{filename}_displacement.txt', delimiter="")[:, 1:]

    stationary = plot_mesh(nodes, elems, title=f"optimized design ({d}um element)")
    displaced = plot_mesh(nodes, elems, disp, sgn*0.1, title=f"first eigenmode ({d}um element)")

    base.savefig(f'../models/accel/{filename}/{filename}_base.pdf')
    stationary.savefig(f'../models/accel/{filename}/{filename}_stationary.pdf')
    displaced.savefig(f'../models/accel/{filename}/{filename}_displaced.pdf')

plt.show()