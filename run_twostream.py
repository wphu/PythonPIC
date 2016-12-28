import numpy as np
from Grid import Grid
from Species import Species
from pic3 import run
import plotting

def two_stream_instability(filename, plasma_frequency=1, qmratio=-1, dt=0.2, NT=300,
                             NG=32, N_electrons=128, L=2 * np.pi, epsilon_0=1,
                             push_amplitude=0.001, push_mode=1, v0=1.0):
    """Implements two stream instability from Birdsall and Langdon"""
    print("Running two stream instability")
    particle_charge = plasma_frequency**2 * L / float(2*N_electrons * epsilon_0 * qmratio)
    particle_mass = particle_charge / qmratio

    g = Grid(L=L, NG=NG, NT=NT)
    k0 = 2*np.pi/g.L
    w0 = plasma_frequency
    print("k0*v0/w0 is", k0*v0/w0, "which means the regime is", "stable" if k0*v0/w0 > 2**0.5 else "unstable")
    electrons1 = Species(particle_charge, particle_mass, N_electrons, "beam1", NT=NT)
    electrons2 = Species(particle_charge, particle_mass, N_electrons, "beam2", NT=NT)
    electrons1.v[:] = v0
    electrons2.v[:] = -v0
    list_species = [electrons1, electrons2]
    for i, species in enumerate(list_species):
        species.distribute_uniformly(g.L, 0.5*g.dx*i)
        species.sinusoidal_position_perturbation(push_amplitude, push_mode, g.L)
    params = NT, dt, epsilon_0
    return run(g, list_species, params, filename)

if __name__ == '__main__':
    two_stream_instability("data_analysis/TS1.hdf5",
                                NG = 32,
                                )
    two_stream_instability("data_analysis/TS2.hdf5",
                                 NG = 64,
                                 )
    two_stream_instability("data_analysis/TS3.hdf5",
                                plasma_frequency=10,
                                N_electrons=int(1e5),
                                )
    two_stream_instability("data_analysis/TS4.hdf5",
                                NT=9000,
                                )
    two_stream_instability("data_analysis/TS5.hdf5",
                                NT=1000,
                                plasma_frequency=1.1*2**-0.5,
                                N_electrons=int(1e5),
                                )
    two_stream_instability("data_analysis/TS6.hdf5",
                                NT=1000,
                                plasma_frequency=1.1*2**-0.5,
                                N_electrons=int(1e5),
                                NG=128,
                                )
    two_stream_instability("data_analysis/TS7.hdf5",
                                NT=1000,
                                plasma_frequency=10,
                                N_electrons=int(1e5),
                                )
    two_stream_instability("data_analysis/TS8.hdf5",
                                NT=1000,
                                plasma_frequency=10,
                                N_electrons=int(1e5),
                                NG=128,
                                )
    two_stream_instability("data_analysis/TS9.hdf5",
                                NT=1000,
                                plasma_frequency=10,
                                N_electrons=int(1e5),
                                NG=256,
                                )
    two_stream_instability("data_analysis/TS10.hdf5",
                                NT=1000,
                                plasma_frequency=10,
                                N_electrons=int(1e5),
                                NG=512,
                                )

    show = False
    plotting.plotting("data_analysis/TS1.hdf5", show=show)
    plotting.plotting("data_analysis/TS2.hdf5", show=show)
    plotting.plotting("data_analysis/TS3.hdf5", show=show)
    plotting.plotting("data_analysis/TS4.hdf5", show=show)
    plotting.plotting("data_analysis/TS5.hdf5", show=show)
    plotting.plotting("data_analysis/TS6.hdf5", show=show)
    plotting.plotting("data_analysis/TS7.hdf5", show=show)
    plotting.plotting("data_analysis/TS8.hdf5", show=show)
    plotting.plotting("data_analysis/TS9.hdf5", show=show)
    plotting.plotting("data_analysis/TS10.hdf5", show=show)
