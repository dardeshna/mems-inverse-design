import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

nodes = np.genfromtxt('../models/beam20p/nodes.txt', delimiter=",")

print(nodes)

elems = [1,    10,    95,    19,    61,   105,   222,   192,     9,    93,    94,    20,   104,   220,   221,   193,    62,   103,   219,   190]

fig = plt.figure()
ax = plt.axes(projection='3d')

nodes_norm = []
for i, e in enumerate(elems):

    ax.scatter3D(*nodes[e-1, 1:])
    ax.text(*nodes[e-1, 1:], i+1, None)
    nodes_norm.append(nodes[e-1, 1:])

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

nodes_norm = np.array(nodes_norm)

for i in range(nodes_norm.shape[-1]):
    nodes_norm[:,i] = (nodes_norm[:,i] - nodes_norm[:,i].min()) / (nodes_norm[:,i].max() - nodes_norm[:,i].min()) * 2 -1

print(np.array(nodes_norm))

print("[", end="")
for i in nodes_norm:
    for j in i:
        print(int(j), end=" ")
    print(";", end=" ")
print("]")

plt.show()
