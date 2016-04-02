import FourierSolver as FS
import MatrixSolver as MS
import numpy as np
import matplotlib.pyplot as plt
N = int(1e5)
NG = 128
NT = 100
L = 1
dt = 0.001
T = NT*dt
particle_charge = -1
particle_mass = 1

def L2norm(A, B):
    return np.sum((A-B)**2)/np.sum(A**2)

x, dx = np.linspace(-L/2,L/2,NG, retstep=True,endpoint=False)
charge_density = np.zeros_like(x)
x_particles = np.linspace(0,L,N, endpoint=False) + L/N/100
x_particles += 0.001*np.sin(x_particles*np.pi/L)
v_particles = np.ones(N)
v_particles[::2] = -1


charge_density = np.sin(2*x*np.pi)
field = np.cos(2*np.pi*x)#/2/np.pi
potential = np.sin(2*np.pi*x)#/4/-np.pi**2


MSmatrix = MS.setup_inverse_matrix(NG, dx)
MSfield, MSpotential = MS.PoissonSolver(charge_density, MSmatrix, dx)

FSfield, FSpotential = FS.PoissonSolver(charge_density, x)

fig, axes = plt.subplots(3)
ax0, ax1, ax2 = axes
ax0.plot(x, charge_density)
ax0.set_title("Charge density")
ax1.set_title("Field")
ax1.plot(x, MSfield, "b-", label="Matrix {:4.2f}".format(L2norm(field, MSfield)))
ax1.plot(x, FSfield, "r-", label="Fourier {:4.2f}".format(L2norm(field, FSfield)))
ax1.plot(x, field, "g-", label="Anal")
ax2.set_title("Potential")
ax2.plot(x, MSpotential, "b-", label="Matrix {:4.2f}".format(L2norm(potential, MSpotential)))
ax2.plot(x, FSpotential, "r-", label="Fourier {:4.2f}".format(L2norm(potential, FSpotential)))
ax2.plot(x, potential, "g-", label="Anal")
for ax in axes:
    ax.grid()
    ax.legend()
plt.show()
