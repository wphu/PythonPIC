import numpy as np

#TODO: include epsilon_0
def setup_inverse_matrix(NG, dx):
    matrix_diagonal = -2*np.ones(NG)
    matrix_offdiagonal = np.ones(NG-1)
    Neumann_FirstOrder_BC = np.zeros((NG-1, NG-1))
    Neumann_FirstOrder_BC[0,0] = Neumann_FirstOrder_BC[-1,-1] = 1
    matrix = np.diag(matrix_diagonal) + np.diag(matrix_offdiagonal, 1) + np.diag(matrix_offdiagonal, -1)
    matrix_inverse = np.linalg.inv(matrix)*(-2*dx)
    return matrix_inverse

def PoissonSolver(charge_density, matrix_inverse):
    potential = matrix_inverse@charge_density
    electric_field = -np.gradient(potential)
    return electric_field, potential
