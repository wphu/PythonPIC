import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.interpolate import interp1d
import FourierSolver

N=int(1e5)
NG = 100
L=1
x_particles=np.array([0.95])
x, dx = np.linspace(0,L,NG, retstep=True,endpoint=False)
print(x.shape)
charge_density = np.zeros_like(x)
matrix_diagonal = -2*np.ones(NG)
matrix_offdiagonal = np.ones(NG-1)
Neumann_FirstOrder_BC = np.zeros((NG-1, NG-1))
Neumann_FirstOrder_BC[0,0] = Neumann_FirstOrder_BC[-1,-1] = 1
matrix = np.diag(matrix_diagonal) + np.diag(matrix_offdiagonal, 1) + np.diag(matrix_offdiagonal, -1)
matrix_inverse = np.linalg.inv(matrix)*(-2*dx)
x_particles = np.linspace(0,L,N, endpoint=False)
# x_particles=np.random.random(N)*L
# x_particles=np.random.normal(L/2, 0.1, N)
particle_charge = -np.ones(N)
particle_mass = np.ones(N)
v_particles = np.zeros(N)#ones(N)
# v_particles[::2]=-1
#charge density deposition
def charge_density_deposition(x, dx, x_particles, particle_charge):
    indices_on_grid = (x_particles/dx).astype(int)
    # print(indices_on_grid)
    # count = np.zeros(10)
    # for i in indices_on_grid:
    #     count[i]+=1
    # print(count)
    # plt.hist(x,x[indices_on_grid])
    # plt.show()
    # plt.hist(indices_on_grid)
    # plt.show()
    #TODO: matricize this

    # charge_density = particle_charge*

    charge_density=np.zeros_like(x)
    for (i, index), xp in zip(enumerate(indices_on_grid), x_particles):
        charge_density[index]+=particle_charge[i] * (dx+x[index]-xp)/dx
        charge_density[(index+1)%(NG)] += particle_charge[i] * (xp - x[index])/dx

    #neutralizing background
    # total_charge = sp.integrate.simps(charge_density, x)
    # charge_per_cell = total_charge/len(x)
    return charge_density - 0#charge_per_cell

charge_density=charge_density_deposition(x, dx, x_particles, particle_charge)

def interpolateField(x_particles, electric_field, x):
    indices_on_grid = (x_particles/dx).astype(int)
    field = (x[indices_on_grid] + dx - x_particles) * electric_field[indices_on_grid] +\
        (x_particles - x[indices_on_grid]) * electric_field[(indices_on_grid+1)%NG]
    return field / dx


def field_quantities(x, charge_density, solver = "fourier"):
    if solver=="matrix":
        potential = matrix_inverse@charge_density
        electric_field = -np.gradient(potential)
    elif solver == "fourier":
        potential, electric_field = FourierSolver.PoissonSolver(charge_density, x, return_potential=True)
    # electric_field_function = sp.interpolate.interp1d(x, electric_field, assume_sorted=True)
    electric_field_function = lambda x_particles: interpolateField(x_particles, electric_field, x)

    return potential, electric_field, electric_field_function

potential, electric_field, electric_field_function = field_quantities(x, charge_density, solver="fourier")

def leapfrog_particle_push(x, v, dt, electric_force):
    v_new = v + electric_force*dt
    return (x + v_new*dt)%L, v_new
dt=0.5

def save_all_the_data():
    return

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

    print(x_particles)
    xhist = position_hist_axes.hist(x_particles,NG, alpha=0.1)
    position_hist_axes.set_ylabel("$N$ at $x$")
    position_hist_axes.set_xlim(0,L)
    position_hist_axes.scatter(x_particles,np.arange(N)/(NG))

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

x_dummy, v_particles = leapfrog_particle_push(x_particles, v_particles, -dt/2., electric_field_function(x_particles)*particle_charge/particle_mass)
for i in range(400):
    print(i)
    all_the_plots(i)
    x_particles, v_particles = leapfrog_particle_push(x_particles,v_particles,dt,electric_field_function(x_particles)*particle_charge/particle_mass)
    charge_density = charge_density_deposition(x, dx, x_particles, particle_charge)
    potential, electric_field, electric_field_function = field_quantities(x, charge_density)
