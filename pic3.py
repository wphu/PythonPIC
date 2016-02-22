import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.interpolate import interp1d
N=int(1e5)
NG = 100
L=1
x, dx = np.linspace(0,L,NG, retstep=True,endpoint=True)


charge_density = np.zeros_like(x)

matrix_diagonal = -2*np.ones(NG)
matrix_offdiagonal = np.ones(NG-1)

Neumann_FirstOrder_BC = np.zeros((NG-1, NG-1))
Neumann_FirstOrder_BC[0,0] = Neumann_FirstOrder_BC[-1,-1] = 1
matrix = np.diag(matrix_diagonal) + np.diag(matrix_offdiagonal, 1) + np.diag(matrix_offdiagonal, -1)
matrix_inverse = np.linalg.inv(matrix)*(-2*dx)

x_particles=np.random.random(N)*L
# x_particles=np.random.normal(L/2, 0.1, N)
particle_charge = np.ones(N)
particle_charge[:N/2] = -1
particle_mass = np.ones(N)
v_particles = np.ones(N)
v_particles[::2]=-1

#charge density deposition
def charge_density_deposition(x, dx, x_particles, particle_charge):
    indices_on_grid = (x_particles//dx).astype(int)
    charge_density=np.zeros_like(x)
    #TODO: matricize this
    for i, index in enumerate(indices_on_grid):
        charge_density[index]+=particle_charge[i]
    return charge_density

charge_density=charge_density_deposition(x, dx, x_particles, particle_charge)

def field_quantities(x, charge_density):
    potential = matrix_inverse@charge_density
    electric_field = -np.gradient(potential)
    electric_field_function = sp.interpolate.interp1d(x, electric_field, bounds_error = False, fill_value=0, assume_sorted=True)
    return potential, electric_field, electric_field_function

potential, electric_field, electric_field_function = field_quantities(x, charge_density)

def leapfrog_particle_push(x, v, dt, electric_force):
    return (x + v*dt)%L, v + electric_force*dt
dt=0.001

def all_the_plots(i):
    # x_particles = np.random.random(100)
    field_particles = electric_field_function(x_particles)
    fig, subplots = plt.subplots(4,2, squeeze=True)
    fig.title("Iteration {}".format(i))
    (charge_axes, d1), (potential_axes, d2), (field_axes, d3), (position_hist_axes, velocity_hist_axes) = subplots
    fig.subplots_adjust(hspace=0)
    charge_axes.plot(x,charge_density, label="charge density")
    potential_axes.plot(x, potential, label="potential")
    field_axes.scatter(x_particles, field_particles, label="interpolated field")
    field_axes.plot(x,electric_field, label="electric field")
    field_axes.set_xlim(0,L)
    velocity_hist_axes.set_xlabel("$x$")
    charge_axes.set_ylabel(r"Charge density $\rho$")
    potential_axes.set_ylabel(r"Potential $V$")
    field_axes.set_ylabel(r"Field $E$")
    position_hist_axes.hist(x_particles,x)
    position_hist_axes.set_ylabel("$N$ at $x$")
    velocity_hist_axes.hist(v_particles,100)
    velocity_hist_axes.set_xlabel("$v$")
    velocity_hist_axes.set_ylabel("$N$ at $v$")
    velocity_hist_axes.set_xlim(-2,2)
    plt.savefig("{}.png".format(i))
    plt.clf()
    plt.close()
for i in range(100):
    print(i)
    all_the_plots(i)
    x_particles, v_particles = leapfrog_particle_push(x_particles,v_particles,dt,electric_field_function(x_particles)*particle_charge/particle_mass)
    charge_density = charge_density_deposition(x, dx, x_particles, particle_charge)
    potential, electric_field, electric_field_function = field_quantities(x, charge_density)
