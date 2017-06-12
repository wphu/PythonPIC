# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.run_laser import laser, impulse_duration, n_macroparticles, plots, number_cells

args = plotting_parser("Hydrogen shield")
perturbation_amplitude = 0
number_particles = 10000
powers = range(23, 20, -1)
power = 23
intensity = 10**power
# for power in powers:
#     intensity = 10**power
for number_particles, n_cells in [
    # [10000, number_cells],
    # [10000, int(number_cells/2)],
    [20000, number_cells],
    [20000, int(2*number_cells)],
    ]:
    s = laser(f"{number_particles}_{n_cells}_run_{power}_{perturbation_amplitude}", number_particles, n_cells, impulse_duration,
              intensity, perturbation_amplitude).lazy_run()
    if any(args):
        plots(s, *args, frames="few")