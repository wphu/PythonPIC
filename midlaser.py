# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.run_laser import laser, impulse_duration, n_macroparticles, plots

args = plotting_parser("Hydrogen shield")
perturbation_amplitude = 0
number_particles = 20000
powers = range(23, 19, -1)
for power in powers:
    intensity = 10**power
    s = laser(f"{number_particles}_run_{power}_{perturbation_amplitude}", number_particles, impulse_duration, intensity, perturbation_amplitude).lazy_run()
    if any(args):
        plots(s, *args, frames="few")
