from numpy import pi

from Runner import Runner
from plotting import plotting


# TODO: shouldn't i be passing a dict here...
def cold_plasma_oscillations(filename, q=-1, m=1, scaling_factor=1, dt=0.2, NT=150,
                             NG=32, N_electrons=128, L=2 * pi, epsilon_0=1, c=1,
                             push_amplitude=0.001, push_mode=1):
    particles = dict(N=N_electrons, q=q * scaling_factor, m=m * scaling_factor, NT=NT, name="particles",
                     initial_position="position_perturbation", mode_number=push_mode, mode_amplitude=push_amplitude)
    run = Runner(NT=NT, dt=dt, epsilon_0=epsilon_0, c=c, NG=NG, L=L, filename=filename, particles=particles)
    run.grid_species_initialization()
    run.run(NT)


if __name__ == '__main__':
    plasma_frequency = 1
    N_electrons = 1024
    epsilon_0 = 1
    qmratio = -1
    L = 2 * pi
    particle_charge = plasma_frequency ** 2 * L / float(N_electrons * epsilon_0 * qmratio)
    particle_mass = particle_charge / qmratio

    cold_plasma_oscillations("data_analysis/CO/CO.hdf5", scaling_factor=1, q=particle_charge, m=particle_mass, NT=150,
                             dt=0.2, NG=64, N_electrons=N_electrons)
    plotting("data_analysis/CO/CO.hdf5", show=False, save=True, animate=False)
