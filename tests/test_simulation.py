# def test_simulation_equality():
#     g = Grid(L=2 * np.pi, NG=32, NT=1)
#     N = 128
#     electrons = Species(-1.0, 1.0, N, "electrons", NT=1)
#     positrons = Species(1.0, 1.0, N, "positrons", NT=1)
#     NT = 100
#     dt = 0.1
#     epsilon_0 = 1
#     date_ver_str = date_version_string()
#
#
#     filename = "test_simulation_data_format.hdf5"
#     if os.path.isfile(filename):
#         os.remove(filename)
#     S = Simulation(NT, dt, epsilon_0, g, [electrons, positrons])
#     S.save_data(filename)
#
#     S_loaded = load_data(filename)
#     assert S == S_loaded
# if __name__ == "__main__":
#     test_simulation_equality()
