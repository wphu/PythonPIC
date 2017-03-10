import numpy as np


qslashm = charge_to_mass_ratio = 1
def BorisPush(x_particles, v_particles, E, B, dt, qslashm):
    """Implements the 1D Boris particle pusher given:
    x_particles: nparray, locations of particles at time t
    v_particles: nparray, velocities of particles at time t-dt/2 (assumes the leapfrog staggering)
    E: nparray, electric field vector interpolated to particle positions
    B: nparray, magnetic field vector same as above
    dt: float, timestep
    """
    half_electric_impulse = qslashm*0.5*E*dt

    #Boris rotation
    v_minus = v_particles + half_electric_impulse
    t = qslashm*dt*0.5*B
    s = 2*t/(1+np.sum(t*t))
    v_prime = v_minus + np.cross(v_minus, t)
    v_plus = v_minus + np.cross(v_prime, s)
    v_final = v_plus + half_electric_impulse

    #standard leapfrog\Euler position push from centered velocity
    x_final = x_particles + v_final*dt

    return x_final, v_final

# Bz = np.ones(5)
# A = np.array([[1, 1],
#             [2,2],
#             [3,3],
#             [4,4],
#             [5,5]])

def Cross2D(A,Bz):
    result = np.empty_like(A)
    result[:,0] = A[:,1]*Bz
    result[:,1] = -A[:,0]*Bz
    return result
# print(Cross2D(A,Bz))

def BorisPush1d2v(x_particles, v_particles, E, B, dt, qslashm):
    """Implements the Boris particle pusher given:
    x_particles: nparray, locations of particles at time t
    v_particles: nparray, velocities of particles at time t-dt/2 (assumes the leapfrog staggering)
    E: nparray, electric field vector interpolated to particle positions
    B: nparray, magnetic field vector same as above
    dt: float, timestep
    """
    half_electric_impulse = qslashm*0.5*E*dt

    #Boris rotation
    v_minus = v_particles + half_electric_impulse
    t = qslashm*dt*0.5*B
    s = 2*t/(1+np.sum(t*t))
    v_prime = v_minus + Cross2D(v_minus, t)
    v_plus = v_minus + Cross2D(v_prime, s)
    v_final = v_plus + half_electric_impulse

    #standard leapfrog\Euler position push from centered velocity
    x_final = x_particles + v_final[:,0]*dt

    return x_final, v_final

if __name__=="__main__":
    import matplotlib.pyplot as plt
    x_particles = np.array([1.])
    v_particles =np.array([[0.,1.]])
    N = 1
    w_omega = qslashm*1
    # dt = 2*np.pi/w_omega
    dt = 0.01
    t = np.arange(0, 100000*dt, dt)
    NT = len(t)
    x_history = np.zeros((NT, N,1))
    v_history = np.zeros((NT, N,2))
    E = np.array([0.,1.])
    B = np.ones_like(x_particles)
    for i in range(NT):
        x_history[i] = x_particles
        v_history[i] = v_particles
        x_particles, v_particles = BorisPush1d2v(x_particles,v_particles, E, B, dt, 1)
    fig, (position_axes, phase_axes, energy_axes) = plt.subplots(3, 1)
    position_axes.plot(t, x_history[:,0,0])
    position_axes.set_xlabel("t")
    position_axes.set_ylabel("x")
    phase_axes.plot(x_history[:,0,0], v_history[:, 0, 0])
    phase_axes.set_xlabel("x")
    phase_axes.set_ylabel("y")
    energy_axes.plot(t, np.sum(v_history**2, axis=2)[:,0])
    energy_axes.set_xlabel("t")
    energy_axes.set_ylabel("v2")
    plt.show()
