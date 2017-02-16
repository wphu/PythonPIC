from numpy import pi

from Runner import Runner, species_args
from plotting import plotting


def cold_plasma_oscillations(filename, q=1, m=-1, scaling_factor=1000, dt=0.2, NT=150,
                             NG=32, N_electrons=128, L=2 * pi, epsilon_0=1, c=1,
                             push_amplitude=0.001, push_mode=1):
    particles = species_args(N_electrons, q * scaling_factor, m * scaling_factor, NT, name="particles",
                             initial_position="uniform")
    run = Runner(NT=NT, dt=dt, epsilon_0=epsilon_0, c=c, NG=NG, L=L, filename=filename, particles=particles)
    print("boo", run.simulation.NT, run.NT)
    run.grid_species_initialization()
    run.run(NT)


if __name__ == '__main__':
    cold_plasma_oscillations("CO.hdf5")
    plotting("CO.hdf5", show=True, save=False)
