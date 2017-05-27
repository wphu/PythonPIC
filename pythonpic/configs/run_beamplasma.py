""" Run two stream instability"""
# coding=utf-8
import numpy as np

from pythonpic.algorithms.helper_functions import plotting_parser
from pythonpic.classes.grid import Grid
from pythonpic.classes.simulation import Simulation
from pythonpic.classes.species import Species
from pythonpic.visualization import plotting


def weakbeam_instability(filename,
                         plasma_frequency=1,
                         qmratio=-1,
                         dt=0.2,
                         T=300 * 0.2,
                         NG=32,
                         N_beam=128,
                         N_plasma=2048,
                         L=2 * np.pi,
                         epsilon_0=1,
                         push_amplitude=0.0001,
                         push_mode=1,
                         v0=0.01,
                         vrandom=0,
                         save_data: bool = True,
                         ):
    """Implements beam-plasma instability from Birdsall and Langdon
    
    * A cold plasma of high density (like in coldplasma)
    * A cold plasma beam of low density injected into the plasma with initial velocity v_0
    
    Plasma frequency of plasma is much higher than that of beam and it's the dominant one
    wave numbers of interest are near k = plasma's frequency / v_0
    
    
    
    
    """
    print("Running beam-plasma instability")
    particle_mass = 1
    particle_charge = particle_mass * qmratio

    def scaling(N):
        return abs(particle_mass * plasma_frequency ** 2 * L / float(
            particle_charge * N * epsilon_0))

    grid = Grid(L=L, NG=NG, T=T)
    filename = f"data_analysis/BP/{filename}/{filename}.hdf5"

    plasma = Species(particle_charge, particle_mass, N_plasma, "plasma", grid.NT, scaling(N_plasma))
    beam = Species(particle_charge, particle_mass, N_beam, "beam2", grid.NT, scaling(N_plasma))
    # total_negative_charge = particle_charge * (N_plasma + N_beam)
    # N_protons = 100
    # q_protons = -total_negative_charge/N_protons
    # proton_mass = 1e16
    beam.v[:, 0] = v0
    plasma.v[:, 0] = 0
    # background = Species(q_protons, proton_mass, N_protons, "protons", NT, scaling(N_plasma))
    # background.v[:,:] = 0
    list_species = [beam, plasma]  # , background]
    for i, species in enumerate(list_species):
        species.distribute_uniformly(L, 0.5 * grid.dx * i)
        species.sinusoidal_position_perturbation(push_amplitude, push_mode, grid.L)
        if vrandom:
            species.random_velocity_perturbation(0, vrandom)
    description = f"Weak beam instability - beam with $v_0$ {v0:.2f} in cold plasma"
    if vrandom:
        description += f" + thermal $v_1$ of standard dev. {vrandom:.2f}"
    description += "\n"
    run = Simulation(grid, list_species, filename=filename, title=description)
    run.grid_species_initialization()
    run.run(save_data)
    return run


def main():
    args = plotting_parser("Weak beam instability")
    np.random.seed(0)
    simulations = [
        weakbeam_instability("BP1",
                             ),
        ]

    for s in simulations:
        plotting.plots(s, *args, alpha=0.5)


if __name__ == '__main__':
    main()
