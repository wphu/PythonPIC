import numpy as np
import scipy.integrate


profiles = {"linear":lambda x: x,
            "quadratic":lambda x: x**2,
            "exponential":lambda x: np.exp(10*(x-1))}


def FDENS(x, moat_left, ramp_length, plasma_length, N, func='linear'):
    func = profiles['linear']
    rectangle_area = (plasma_length - ramp_length)
    modified_func = lambda x: func((x - moat_left) / ramp_length)
    ramp_area = scipy.integrate.quad(modified_func, moat_left, moat_left + ramp_length)
    triangle_area = 0.5 * ramp_length
    normalization = N / (rectangle_area + triangle_area)
    result = np.zeros_like(x)
    region1 = x < moat_left
    region2 = (x < moat_left + ramp_length) & ~region1
    region3 = (x < moat_left + plasma_length) & ~(region2 | region1)
    result[region2] = normalization * modified_func(x[region2])
    result[region3] = normalization
    return result

