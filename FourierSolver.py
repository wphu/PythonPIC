import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt

def PoissonSolver(rho, x, epsilon_0 = 1):
    #TODO: Aliasing: keep abs(delta_x k) < pi
    #TODO: read arsenous blog
    NG = len(x)
    dx = x[1]-x[0]
    rho_F = fft.fft(rho)
    rho_F[0] = 0
    k = NG*dx*fft.fftfreq(NG,dx)
    k[0] = 0.0001
    field_F = rho_F/(np.pi*2j*k * epsilon_0)
    potential_F = field_F/(-2j*np.pi*k * epsilon_0)
    field = fft.ifft(field_F).real
    potential = fft.ifft(potential_F).real
    energy = np.abs((0.5*np.sum(
        (rho_F*potential_F.conjugate())[:NG//2]
        )))
    # import ipdb; ipdb.set_trace()
    # field -= field.ptp()/2
    # potential -= potential.ptp()/2
    return field, potential, energy
