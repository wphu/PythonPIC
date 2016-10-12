import numpy as np
import matplotlib.pyplot as plt
import FourierSolver
import Simulation
import diagnostics
from constants import epsilon_0
from parameters import NT, NG, N, T, dt, particle_mass, particle_charge, L,x, dx, x_particles, v_particles
import argparse
import time


def charge_density_deposition(x, dx, x_particles, particle_charge):
    """Calculates the charge density on a 1D grid given an array of charged particle positions.
    x: array of grid positions
    dx: grid positions step
    x_particles: array of particle positions on the grid.
        make sure this is 0 < x_particles < L
    """
    assert ((x_particles<L).all() and (0<=x_particles).all()), (x_particles, x_particles[x_particles>L])
    indices_on_grid = (x_particles/dx).astype(int)

    charge_density=np.zeros_like(x)
    for (i, index), xp in zip(enumerate(indices_on_grid), x_particles):
        charge_density[index]+=particle_charge * (dx+x[index]-xp)/dx
        charge_density[(index+1)%(NG)] += particle_charge * (xp - x[index])/dx
    return charge_density


def interpolateField(x_particles, electric_field, x, dx):
    #TODO: test this function, see how it behaves at boundaries
    #TODO: see birdsall 40
    #TODO: implement cubic spline interpolation
    indices_on_grid = (x_particles/dx).astype(int)
    field = (x[indices_on_grid] + dx - x_particles) * electric_field[indices_on_grid] +\
        (x_particles - x[indices_on_grid]) * electric_field[(indices_on_grid+1)%NG]
    return field / dx

def field_quantities(x, charge_density):
    #TODO: this is neither elegant nor efficient :( can probably be rewritten)
    dx = x[1] - x[0]
    potential, electric_field, fourier_field_energy = FourierSolver.PoissonSolver(charge_density, x)
    electric_field_function = lambda x_particles: interpolateField(x_particles, electric_field, x, dx)
    return potential, electric_field, electric_field_function, fourier_field_energy

def leapfrog_particle_push(x, v, dt, electric_force):
    #TODO: make sure energies are given at proper times (at same time for position, velocity)
    #TODO: make sure omega_zero * dt <= 2 to remove errors
    v_new = v + electric_force*dt
    return (x + v_new*dt)%L, v_new

def all_the_plots(i):
    # x_particles = np.random.random(100)
    field_particles = electric_field_function(x_particles)
    fig, subplots = plt.subplots(3,2, squeeze=True)
    (charge_axes, d1), (field_axes, d3), (position_hist_axes, velocity_hist_axes) = subplots
    fig.subplots_adjust(hspace=0)

    charge_axes.plot(x,charge_density, label="charge density")
    charge_axes.plot(x, potential, "g-")
    charge_axes.scatter(x, np.zeros_like(x))
    charge_axes.set_xlim(0,L)
    charge_axes.set_ylabel(r"Charge density $\rho$, potential $V$")

    position_hist_axes.hist(x_particles,NG, alpha=0.1)
    position_hist_axes.set_ylabel("$N$ at $x$")
    position_hist_axes.set_xlim(0,L)

    field_axes.set_ylabel(r"Field $E$")
    field_axes.scatter(x_particles, field_particles, label="interpolated field")
    field_axes.plot(x,electric_field, label="electric field")
    field_axes.set_xlim(0,L)



    velocity_hist_axes.set_xlabel("$x$")
    velocity_hist_axes.hist(np.abs(v_particles),100)
    velocity_hist_axes.set_xlabel("$v$")
    velocity_hist_axes.set_ylabel("$N$ at $v$")
    d1.scatter(x_particles, v_particles)
    d1.set_xlim(0,L)
    plt.savefig("{:03d}.png".format(i))
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    # plt.show()
    plt.clf()
    plt.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="hdf5 file name for storing data")
    args = parser.parse_args()
    if(args.filename[-5:] != ".hdf5"):
        args.filename = args.filename + ".hdf5"

        S = Simulation.Simulation(NT, NG, N, T, particle_charge, particle_mass, L, epsilon_0)
    charge_density=charge_density_deposition(x, dx, x_particles, particle_charge)

    potential, electric_field, electric_field_function, fourier_field_energy = field_quantities(x, charge_density)


    x_dummy, v_particles = leapfrog_particle_push(x_particles, v_particles, -dt/2., electric_field_function(x_particles)*particle_charge/particle_mass)
    kinetic, field, total = 0, 0, 0

    start_time = time.time()
    for i in range(NT):
        S.update_grid(i, charge_density, electric_field)
        S.update_particles(i, x_particles, v_particles)

        kinetic, field, total =diagnostics.energies(x_particles,v_particles,particle_mass,dx, potential, charge_density)
        print("i{:4d} T{:12.3e} V{:12.3e} E{:12.3e}".format(i, kinetic, field, total))

        x_particles, v_particles = leapfrog_particle_push(x_particles,v_particles,dt,electric_field_function(x_particles)*particle_charge/particle_mass)
        charge_density = charge_density_deposition(x, dx, x_particles, particle_charge)
        potential, electric_field, electric_field_function, fourier_field_energy = field_quantities(x, charge_density)

        diag = kinetic, fourier_field_energy, kinetic + fourier_field_energy
        S.update_diagnostics(i, diag)

    runtime = time.time()-start_time
    print("Runtime: {}".format(runtime))
    S.save_data(filename=args.filename)
