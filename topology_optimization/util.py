import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_mesh(nodes, elems, displacement=None, scale=1, color='orange'):

    c = 1/2 * (np.min(nodes, axis=0) + np.max(nodes, axis=0))
    d = np.max(np.max(nodes, axis=0) - np.min(nodes, axis=0))

    if displacement is not None:
        nodes += displacement / np.max(displacement) * d * scale

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    s = np.array((
        (0,1,2,3),
        (4,5,6,7),
        (0,1,5,4),
        (1,2,6,5),
        (2,3,7,6),
        (3,0,4,7),
    ))

    for e in elems:

        coords = nodes[e]
        p = Poly3DCollection(coords[s], ec='black', fc=color)
        ax.add_collection3d(p)

    ax.set_xlim(c[0] - 0.7*d, c[0] + 0.7*d)
    ax.set_ylim(c[1] - 0.7*d, c[1] + 0.7*d)
    ax.set_zlim(c[2] - 0.7*d, c[2] + 0.7*d)

    return fig
    

if __name__ == "__main__":

    from models.accel.gen_accel_mesh import AccelBlankMeshGenerator

    g = AccelBlankMeshGenerator(5200, 2400, 320, 69, 400)
    plot_mesh(g.nodes, g.elems, None, None)
    plt.show()