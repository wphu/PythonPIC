import numpy as np


def l2_norm(reference, test):
    return np.sum((reference - test)**2) / np.sum(reference**2)


def l2_test(reference, test, rtol=1e-3):
    norm = l2_norm(reference, test)
    print("L2 norm: ", norm)
    return norm < rtol
