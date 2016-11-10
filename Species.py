import numpy as np


class Species(object):
    def __init__(self, q, m, N, name=None):
        self.q = q
        self.m = m
        self.N = int(N)
        self.x = np.zeros(N, dtype=float)
        self.v = np.zeros(N, dtype=float)
        if name:
            self.name = name
        else:
            self.name = "q{}m{}N{}".format(q, m, N)

    def leapfrog_init(self, electric_field_function, dt):
        """Leapfrog pusher initialization
        dt: usual timestep, minus halving is done automatically"""

        electric_force = electric_field_function(self.x) * self.q / self.m
        v_new = self.v - electric_force * 0.5 * dt
        energy = self.v * v_new * (0.5 * self.m)
        self.v = v_new
        return energy

    def push_particles(self, electric_field_function, dt, L):
        """Leapfrog pusher"""
        electric_force = electric_field_function(self.x) * self.q / self.m
        v_new = self.v + electric_force * dt

        self.x += v_new * dt
        self.x %= L
        energy = self.v * v_new * (0.5 * self.m)
        self.v = v_new
        return energy

    def boris_push_particles(self, electric_field_function, magnetic_field_function,  dt, L):
        """Boris pusher, unrelativistic"""
        # add half electric impulse to v(t-dt/2)
        vminus = self.v + self.q * electric_field_function(self.x) / self.m * dt * 0.5

        # rotate to add magnetic field
        t = -magnetic_field_function(self.x) * self.q / self.m * dt * 0.5
        s = 2*t/(1+t*t)

        vprime = vminus + np.cross(vminus, t) # TODO: axis?
        vplus = vminus + np.cross(vprime, s)
        v_new = vplus + self.q * electric_field_function(self.x) / self.m * dt * 0.5

        self.x += v_new[:,0] * dt

        self.x %= L
        energy = self.v * v_new * (0.5 * self.m)
        self.v = v_new
        return energy
        # add remaining half of electric impulse


    def distribute_uniformly(self, Lx, shift=False):
        self.x = (np.linspace(0, Lx, self.N, endpoint=False) + shift * self.N / Lx / 10) % Lx

    def sinusoidal_position_perturbation(self, amplitude, mode, L):
        self.x += amplitude * np.cos(2 * mode * np.pi * self.x / L)
        self.x %= L

    def __eq__(self, other):
        result = True
        result *= self.q == other.q
        result *= self.m == other.m
        result *= self.N == other.N
        result *= np.isclose(self.x, other.x).all()
        result *= np.isclose(self.v, other.v).all()
        result *= self.name == other.name
        return result

if __name__=="__main__":
    import matplotlib.pyplot as plt
    particles = Species(1, 1, 1, "particle")
    particles.x = np.array([5], dtype=float)
    particles.v = np.array([[0,1,0]], dtype=float)

    x_history = []
    v_history = []
    T, dt = np.linspace(0, 10, 1000, retstep=True)
    for t in T:
        particles.boris_push_particles(lambda x: np.zeros_like(x), lambda x: np.array([[0,0,1]]), dt, np.inf)
        x_history.append(particles.x[0])
        v_history.append(particles.v[0,0])

    plt.plot(x_history, v_history)
    plt.show()
