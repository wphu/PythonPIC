# """Tests relativistic species"""
# # coding=utf-8
# import matplotlib.pyplot as plt
# import numpy as np
#
# from Constants import Constants
# from Simulation import Simulation
# from Species import RelativisticSpecies
# from static_plots import velocity_time_plots
#
#
# def setup(NT, dt, epsilon_0, c, e_field_magnitude, b_field_magnitude, vx=0, vy=0, vz=0):
#     q = 1
#     m = 1
#     N = 1
#     name = "test particles"
#     s = RelativisticSpecies(q, m, N, name, NT)
#     s.distribute_uniformly(1)
#     s.v[:, 0] = vx
#     s.v[:, 1] = vy
#     s.v[:, 2] = vz
#     simulation = Simulation(NT, dt, Constants(epsilon_0, c), None, [s])
#
#     def electric_field_function(x):
#         result = np.zeros((N, 3))
#         result[:, 0] = e_field_magnitude
#         return result
#
#     def magnetic_field_function(x):
#         result = np.zeros((N, 3))
#         result[:, 2] = b_field_magnitude
#         return result
#
#     for i in range(NT):
#         s.save_particle_values(i)
#         kinetic_energy = s.push(electric_field_function, dt, magnetic_field_function, c)
#
#     return simulation
#
#
# def test_uniform_electric():
#     dt = 1e-2
#     NT = 10000
#     simulation = setup(NT, dt, 8.854e-12, 3e8, e_field_magnitude=1, b_field_magnitude=0)
#     s = simulation.list_species[0]
#     t = np.arange(NT) * dt
#     estimated_field_magnitude = np.polyfit(t, s.velocity_history[:, 0, 0], 2)[1]
#     print(estimated_field_magnitude)
#     fig = velocity_time_plots(s, dt)
#     assert np.isclose(estimated_field_magnitude, 1), plt.show()
#
#
# def test_circular():
#     dt = 1e-2
#     NT = 10000
#     simulation = setup(NT, dt, 8.854e-12, 3e8, e_field_magnitude=0, b_field_magnitude=1, vx=1)
#     s = simulation.list_species[0]
#     t = np.arange(NT) * dt
#     fig = velocity_time_plots(s, dt)
#     plt.show()
#     assert np.isclose(estimated_field_magnitude, 1)
#
#
# def test_circular_relativistic():
#     dt = 1e-2
#     NT = 10000
#     simulation = setup(NT, dt, 8.854e-12, 3e8, e_field_magnitude=0, b_field_magnitude=1, vx=4e8)
#     s = simulation.list_species[0]
#     t = np.arange(NT) * dt
#     fig = velocity_time_plots(s, dt)
#     plt.show()
#     assert np.isclose(estimated_field_magnitude, 1), plt.show()
#
#
# if __name__ == '__main__':
#     test_circular()
