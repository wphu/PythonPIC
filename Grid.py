import numpy as np
from FourierSolver import PoissonSolver
from scatter import charge_density_deposition
from gather import interpolateField
from constants import epsilon_0


class Grid(object):
    def __init__(self, L=2 * np.pi, NG=32):
        self.x, self.dx = np.linspace(0, L, NG, retstep=True, endpoint=False)
        self.charge_density = np.zeros_like(self.x)
        self.electric_field = np.zeros_like(self.x)
        self.potential = np.zeros_like(self.x)
        self.L = L
        self.NG = int(NG)

    def solve_poisson(self):
        self.electric_field, self.potential, field_energy = PoissonSolver(self.charge_density, self.x, epsilon_0=epsilon_0)
        return field_energy

    def gather_charge(self, species):
        self.charge_density = charge_density_deposition(self.x, self.dx, species.x, species.q)

    def electric_field_function(self, xp):
        return interpolateField(xp, self.electric_field, self.x, self.dx)

    # def plot(self, show=True):
    #     plt.plot(self.x, self.charge_density
