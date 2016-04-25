import numpy as np
import h5py

class Species(object):
    def __init__(self, name, q, m, N):
        self.attrs = {"name": name, "q": q, "m": m, "N": N}
        self.q = q
        self.m = m
        self.N = N
        self.r = np.zeros(N, dtype=float)
        self.v = np.zeros(N, dtype=float)
    def distribute_uniformly(self, Lx, shift):
        self.r = (np.linspace(0, Lx, self.N, endpoint = False) + shift) % Lx
    def interpolate_scalar(self, scalar_field, dx):
        indices = (self.r // dx).astype(int)
        positions_in_cells = self.r % dx
        fractions_to_right = positions_in_cells/dx
        fractions_to_left = 1 - fractions_to_right

        return scalar_field[indices]*fractions_to_left + scalar_field[(indices+1)%len(scalar_field)]*fractions_to_right
    # def export_to_hdf5(self,)
if __name__=="__main__":
    import matplotlib.pyplot as plt

    electrons = Species("electrons", -1, 1, 10)
    Lx = 1
    grid_original, dx = np.linspace(0, Lx, 6, retstep=True, endpoint=False)
    grid = grid_original**2
    electrons.distribute_uniformly(Lx, Lx/100)
    plt.plot(grid_original, grid, "bo-")
    plt.plot(electrons.r, electrons.interpolate_scalar(grid, dx), "ro")
    print(electrons.r)
    print(grid)
    print(electrons.interpolate_scalar(grid, dx))
    print((np.linspace(0,Lx, electrons.N, endpoint=False) + Lx/100))
    plt.show()
