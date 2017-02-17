from numpy import pi

from Runner import Runner, species_args
from plotting import plotting


def cold_plasma_oscillations(filename, q=-1, m=1, scaling_factor=1, dt=0.1, NT=150,
                             NG=32, N_electrons=128, L=2 * pi, epsilon_0=1, c=1,
                             push_amplitude=0.001, push_mode=1):
    particles = species_args(N_electrons, q * scaling_factor, m * scaling_factor, NT, name="particles",
                             initial_position="uniform")
    run = Runner(NT=NT, dt=dt, epsilon_0=epsilon_0, c=c, NG=NG, L=L, filename=filename, particles=particles)
    run.grid_species_initialization()
    run.run(NT)


if __name__ == '__main__':
    plasma_frequency = 1
    N_electrons = 128
    epsilon_0 = 1
    qmratio = 1
    L = 2 * pi
    particle_charge = plasma_frequency ** 2 * L / float(N_electrons * epsilon_0 * qmratio)

    cold_plasma_oscillations("data_analysis/CO/CO.hdf5", scaling_factor=particle_charge)
    plotting("data_analysis/CO/CO.hdf5", show=False, save=True, animate=False)
