""" Run cold plasma oscillations"""
# coding=utf-8
import numpy as np
from numpy import pi

from Constants import Constants
from Grid import Grid
from Simulation import Simulation
from Species import Species
from plotting import plotting
from helper_functions import plotting_parser, get_dominant_mode


def cold_plasma_oscillations(filename,
                             plasma_frequency=1,
                             qmratio=-1,
                             dt: float = 0.2,
                             NT: int = 150,
                             NG: int = 32,
                             N_electrons: int = 128,
                             L: float = 2 * pi,
                             epsilon_0: float = 1,
                             c: float = 1,
                             push_amplitude: float = 0.001,
                             push_mode: float = 1,
                             save_data: bool = True):
    """
    Runs cold plasma oscilltaions

    :param str filename: hdf5 file name
    :param float q: particle charge
    :param float m: particle mass
    :param float scaling: how many particles should be represented by each superparticle
    :param float dt: timestep
    :param int NT: number of timesteps to run
    :param int N_electrons: number of electron superparticles
    :param int NG: number of cells on grid
    :param float L: grid size
    :param float epsilon_0: the physical constant
    :param float c: the speed of light
    :param float push_amplitude: amplitude of initial position displacement
    :param int push_mode: mode of initially excited mode
    :param bool save_data: 
    """

    filename = f"data_analysis/CO/{filename}/{filename}.hdf5"
    particle_mass = 1
    particle_charge = particle_mass * qmratio
    scaling = abs(particle_mass * plasma_frequency ** 2 * L / float(
        particle_charge * N_electrons * epsilon_0))

    particles = Species(N=N_electrons, q=particle_charge, m=particle_mass, name="electrons", NT=NT, scaling=scaling)
    particles.distribute_uniformly(L)
    particles.sinusoidal_position_perturbation(push_amplitude, push_mode, L)
    grid = Grid(L, NG, epsilon_0, NT)

    description = f"Cold plasma oscillations\nposition initial condition perturbed by sinusoidal oscillation mode {push_mode} excited with amplitude {push_amplitude}\n"

    run = Simulation(NT, dt, Constants(c, epsilon_0), grid, [particles], filename=filename,
                     title=description)
    run.grid_species_initialization()
    run.run(save_data)
    return run

if __name__ == '__main__':
    plasma_frequency = 1
    push_mode = 2
    N_electrons = 1024
    NG = 64
    qmratio = -1

    S = cold_plasma_oscillations(f"CO1", qmratio=qmratio, plasma_frequency=plasma_frequency, NG=NG,
                                 N_electrons=N_electrons, push_mode=push_mode, save_data=False)
    show, save, animate = plotting_parser("Cold plasma oscillations")
    plotting(S, show=show, save=save, animate=animate)