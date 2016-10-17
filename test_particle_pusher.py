import particle_pusher

def test_polynomial_field():
    NG = 32
    L = 1
    x, dx = np.linspace(0, L, NG, retstep=True, endpoint=False)

    x_particles = np.array([L/2]
    v_particles = np.zeros(1)
