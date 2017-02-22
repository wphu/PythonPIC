""" Run cold plasma oscillations"""
# coding=utf-8
from numpy import pi

from Runner import Runner
from plotting import plotting


# TODO: shouldn't i be passing a dict here...
def cold_plasma_oscillations(filename: str, q: float = -1, m: float = 1, scaling_factor: float= 1, dt: float = 0.2, NT: int = 150,
                             NG: int = 32, N_electrons: int = 128, L: float = 2 * pi, epsilon_0: float = 1, c: float = 1,
                             push_amplitude: float = 0.001, push_mode: float = 1):
    """
    Runs cold plasma oscilltaions

    :param str filename: hdf5 file name
    :param float q: particle charge
    :param float m: particle mass
    :param float scaling_factor: how many particles should be represented by each superparticle
    :param float dt: timestep
    :param int NT: number of timesteps to run
    :param int N_electrons: number of electron superparticles
    :param int NG: number of cells on grid
    :param float L: grid size
    :param float epsilon_0: the physical constant
    :param float c: the speed of light
    :param float push_amplitude: amplitude of initial position displacement
    :param int push_mode: mode of initially excited mode
    """
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

    cold_plasma_oscillations("data_analysis/CO/CO.hdf5", q=particle_charge, m=particle_mass, NG=64,
                             N_electrons=N_electrons)
    plotting("data_analysis/CO/CO.hdf5", show=False, save=True, animate=False)

