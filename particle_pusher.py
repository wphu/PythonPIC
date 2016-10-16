def leapfrog_particle_push(x, v, dt, electric_force, L):
    """the most basic of particle pushers"""
    # TODO: make sure energies are given at proper times (at same time for position, velocity)
    # TODO: make sure omega_zero * dt <= 2 to remove errors
    v_new = v + electric_force * dt
    return (x + v_new * dt) % L, v_new
