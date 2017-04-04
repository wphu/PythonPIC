""" Run two stream instability"""
# coding=utf-8
import numpy as np

import plotting
from Constants import Constants
from Grid import Grid
from Simulation import Simulation
from Species import Species
from helper_functions import plotting_parser


def weakbeam_instability(filename,
                         plasma_frequency=1,
                         qmratio=-1,
                         dt=0.2,
                         NT=300,
                         NG=32,
                         N_beam=128,
                         N_plasma=2048,
                         L=2 * np.pi,
                         epsilon_0=1,
                         push_amplitude=0.01,
                         push_mode=1,
                         v0=1.0,
                         vrandom=0,
                         save_data: bool = True,
                         ):
    """Implements beam-plasma instability from Birdsall and Langdon
    
    * A cold plasma of high density (like in coldplasma)
    * A cold plasma beam of low density injected into the plasma with initial velocity v_0
    
    Plasma frequency of plasma is much higher than that of beam and it's the dominant one
    wave numbers of interest are near k = plasma's frequency / v_0
    
    
    
    
    """
    print("Running two stream instability")
    particle_mass = 1
    particle_charge = particle_mass * qmratio

    def scaling(N):
        return abs(particle_mass * plasma_frequency ** 2 * L / float(
            particle_charge * N * epsilon_0))

    filename = f"data_analysis/BP/{filename}/{filename}.hdf5"

    grid = Grid(L=L, NG=NG, NT=NT, n_species=2)
    plasma = Species(particle_charge, particle_mass, N_plasma, "plasma", NT, scaling(N_plasma))
    beam = Species(particle_charge, particle_mass, N_beam, "beam2", NT, scaling(N_plasma))
    beam.v[:] = v0
    plasma.v[:] = 0
    list_species = [beam, plasma]
    for i, species in enumerate(list_species):
        species.distribute_uniformly(L, 0.5 * grid.dx * i)
        species.sinusoidal_position_perturbation(push_amplitude, push_mode, grid.L)
        if vrandom:
            species.random_velocity_perturbation(0, vrandom)
    description = f"Weak beam instability - beam with $v_0$ {v0:.2f} in cold plasma"
    if vrandom:
        description += f" + thermal $v_1$ of standard dev. {vrandom:.2f}"

    run = Simulation(NT, dt, Constants(1, epsilon_0),
                     grid, list_species, filename=filename, title=description)
    run.grid_species_initialization()
    run.run(save_data)
    return run


if __name__ == '__main__':
    np.random.seed(0)
    simulations = [
        weakbeam_instability("BP1",
                             ),
        ]

    show, save, animate = plotting_parser("Weak beam instability")
    for s in simulations:
        plotting.plotting(s, show=show, alpha=0.5, save=save, animate=animate)
