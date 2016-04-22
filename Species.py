import numpy as np
import h5py

class Species(object):
    def __init__(self, name, q, m, N):
        self.attrs = {"name": name, "q": q, "m": m, "N":N}
        self.r = np.zeros(N, dtype=float)
        self.v = np.zeros(N, dtype=float)
if __name__=="__main__":
    electrons = Species("electrons", -1, 1, 1000)
    print(electrons.attrs)
    print(electrons.r)
