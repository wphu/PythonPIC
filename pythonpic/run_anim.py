from pythonpic.configs import cold_plasma_oscillations, wave_propagation
from pythonpic.visualization import plotting

S = cold_plasma_oscillations("test_animation_2")
plotting.plots(S, False, False, True, False)
S = wave_propagation(("test_animation_wave"))
plotting.plots(S, False, False, True, False)
