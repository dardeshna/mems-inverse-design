import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

class AccelBlankMeshGenerator():

    def __init__(self, total_length, proof_mass_length, proof_mass_thickness, suspension_thickness, element_length):

        self.L = total_length
        self.l = proof_mass_length
        self.T = proof_mass_thickness
        self.t = suspension_thickness
        self.d = element_length

        self.generate()

    @property
    def centers(self):

        return np.c_[self.cent_x, self.cent_y, self.cent_z]

    @property
    def nodes(self):

        return np.c_[self.x, self.y, self.z]

    def remove_elems(self, elems_to_remove):

        old_nodes = np.max(self.elems)+1

        idx_keep = np.ones(self.elems.shape[0], dtype=bool)
        idx_keep[elems_to_remove] = False

        idx_domain = np.zeros(self.elems.shape[0], dtype=bool)
        idx_domain[self.domain] = True

        self.domain = np.where((idx_domain & idx_keep)[idx_keep])[0]
        
        self.cent_x = self.cent_x[idx_keep]
        self.cent_y = self.cent_y[idx_keep]
        self.cent_z = self.cent_z[idx_keep]       

        self.elems = self.elems[idx_keep]

        used_nodes = np.unique(self.elems)
        num_used = used_nodes.shape[0]

        self.x = self.x[used_nodes]
        self.y = self.y[used_nodes]
        self.z = self.z[used_nodes]

        self.indices = np.arange(num_used)
        new_idx = np.zeros(old_nodes, dtype=int)
        new_idx[used_nodes] = self.indices

        self.elems = new_idx[self.elems]
        self.fixed = new_idx[np.intersect1d(self.fixed, used_nodes)]

    def generate(self):

        layers = int(2*np.floor(self.T/self.t/2)+1)
        width = self.L//self.d

        n_xy = width*2+1
        n_z = layers*2+1

        nums = np.arange(n_xy*n_xy*n_z).reshape((n_z, n_xy, n_xy)).T

        elems = np.zeros((width, width, layers, 20), dtype=int)
        cent = np.zeros((width, width, layers), dtype=int)

        idx = tuple(map(tuple, np.array([
                [-1, -1, -1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [-1,  1, -1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1,  1,  1],
                [-1,  1,  1],
                [ 0, -1, -1],
                [ 1,  0, -1],
                [ 0,  1, -1],
                [-1,  0, -1],
                [ 0, -1,  1],
                [ 1,  0,  1],
                [ 0,  1,  1],
                [-1,  0,  1],
                [-1, -1,  0],
                [ 1, -1,  0],
                [ 1,  1,  0],
                [-1,  1,  0],
            ]).T+1))

        for i in range(width):
            for j in range(width):
                for k in range(layers):
                    sub = nums[2*i:2*i+3, 2*j:2*j+3, 2*k:2*k+3]
                    elems[i, j, k] = sub[idx]
                    cent[i, j, k] = sub[1,1,1]

        used_nodes = np.unique(elems)
        num_used = used_nodes.shape[0]

        x = np.mod(used_nodes, n_xy) * self.d/2
        y = np.mod(used_nodes // n_xy, n_xy) * self.d/2

        other_heights = (self.T-self.t) / (layers-1)

        heights = np.cumsum(np.array([0,] + [other_heights/2,]*(layers-1) + [self.t/2,]*2 + [other_heights/2,]*(layers-1)))

        z = heights[used_nodes // (n_xy * n_xy)]

        indices = np.arange(num_used)
        new_idx = np.zeros((n_xy*n_xy*n_z), dtype=int)
        new_idx[used_nodes] = indices

        elems = new_idx[elems.reshape(-1, 20)]
        cent = cent.flatten()

        cent_x = np.mod(cent, n_xy) * self.d/2
        cent_y = np.mod(cent // n_xy, n_xy) * self.d/2
        cent_z = heights[2 * (np.arange(elems.shape[0]) % layers) + 1]

        in_mass = ((cent_x > (self.L-self.l)/2) & (cent_x < (self.L-self.l)/2+self.l) & (cent_y > (self.L-self.l)/2) & (cent_y < (self.L-self.l)/2+self.l))
        in_middle = z[elems[:,-1]] == self.T/2
        in_model = in_middle | in_mass

        self.domain = np.where((in_middle & ~in_mass)[in_model])[0]
        elems = elems[in_model]

        used_nodes = np.unique(elems)
        num_used = used_nodes.shape[0]
        x, y, z = x[used_nodes], y[used_nodes], z[used_nodes]

        self.indices = np.arange(num_used)
        new_idx.fill(0)
        new_idx[used_nodes] = self.indices

        self.elems = new_idx[elems]

        self.fixed = self.indices[(x == 0) | (x == self.L) | (y == 0) | (y == self.L)]

        scale = 1e-3
        self.x,self.y,self.z = scale*x, scale*y, scale*z
        self.cent_x,self.cent_y,self.cent_z = scale*cent_x[in_model], scale*cent_y[in_model], scale*cent_z[in_model]

    def dump_txt(self, filename="accel_blank"):

        dir = os.path.dirname(os.path.join(base_dir, f'{filename}.txt'))
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        np.savetxt(
            os.path.join(base_dir, f'{filename}{"_" if filename else ""}elems.txt'),
            np.c_[np.arange(self.elems.shape[0])+1, self.elems+1], fmt="%i", delimiter=", ",
        )
        np.savetxt(
            os.path.join(base_dir, f'{filename}{"_" if filename else ""}nodes.txt'),
            np.c_[self.indices+1, self.x, self.y, self.z], fmt="%i, %e, %e, %e",
        )
        np.savetxt(
            os.path.join(base_dir, f'{filename}{"_" if filename else ""}fixed.txt'),
            self.fixed[None,...]+1, fmt="%i", delimiter=", "
        )

    def dump_inp(self, E, nu, rho, filename="accel_blank"):

        dir = os.path.dirname(os.path.join(base_dir, f'{filename}.txt'))
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        with open(os.path.join(base_dir, f'{filename}.inp'), 'w') as f:
            f.write("*NODE, NSET=Nall\n")

            np.savetxt(f, np.c_[self.indices+1, self.x, self.y, self.z], fmt="%i, %e, %e, %e")

            f.write("\n*ELEMENT, TYPE=C3D20, ELSET=Eall\n")

            for i, e in enumerate(self.elems+1):
                f.write(str(i+1) + ", ")
                for j in e[:9]:
                    f.write(str(j) + ", ")
                f.write(str(e[9]) + ",\n\t" + str(e[10]) + ", ")
                for j in e[11:-1]:
                    f.write(str(j) + ", ")
                f.write(str(e[-1]) + "\n")

            f.write("\n*NSET, NSET=Nfixed\n")

            fixed = self.fixed+1

            for i in range(fixed.shape[0] // 10 - (not bool(fixed.shape[0] % 10))):
                e = fixed[10*i:10*(i+1)]
                for j in e[:-1]:
                    f.write(str(j) + ", ")
                f.write(str(e[-1]) + ",\n")

            e = fixed[-(fixed.shape[0] % 10 if fixed.shape[0] % 10 != 0 else 10):]
            for j in e[:-1]:
                f.write(str(j) + ", ")
            f.write(str(e[-1]) + "\n")

            f.write(f"""\n*BOUNDARY
Nfixed, 1
*BOUNDARY
Nfixed, 2
*BOUNDARY
Nfixed, 3
*MATERIAL,NAME=EL
*ELASTIC
{E}, {nu}
*DENSITY
{rho}
*SOLID SECTION,ELSET=EALL,MATERIAL=EL
*STEP
*FREQUENCY
10
*NODE FILE
U
*EL FILE
S, E
*END STEP""")

            f.close()

if __name__ == "__main__":
    g = AccelBlankMeshGenerator(5200, 2400, 320, 69, 400)
    g.dump_inp(E=170000, nu=0.280, rho=2.329e-09)
    g.dump_txt()