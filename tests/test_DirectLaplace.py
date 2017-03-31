import pytest
import matplotlib.pyplot as plt
import numpy as np
from algorithms_grid import DirectLaplaceSolver

# TODO: use of multigrid methods for wave equation
@pytest.mark.parametrize(["N", "c", "dx"], [(100, 100, 0.2)])
def test_convergence(N, c, dx):
    #dx/dt = c -> dt = dx/c
    dt = dx/c
    potential = np.zeros(N)
    potential = np.sin(np.pi * np.arange(N)/N)
    potential[0] = 1.0
    initial_potential = potential.copy()
    plt.plot(initial_potential)
    # for n in range(200*int(1/dt)):
    for n in range(1000):
        potential = DirectLaplaceSolver(potential, c, dx)
        if n % 100: plt.plot(potential)
    plt.show()
    assert False

if __name__ == "__main__":
    test_convergence(100, 100, 0.2)
