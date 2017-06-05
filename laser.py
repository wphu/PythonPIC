# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.run_laser import laser, impulse_duration, n_macroparticles, plots

args = plotting_parser("Hydrogen shield")
perturbation_amplitude = 0
number_particles = 10000
for power in range(21, 24):
    intensity = 10**power
    s = laser(f"{number_particles}_run_{power}_{perturbation_amplitude}", number_particles, impulse_duration, intensity, perturbation_amplitude).lazy_run()
    plots(s, *args, frames="few")
for power in range(21, 24):
    intensity = 10**power
    s = laser(f"production_run_{power}_{perturbation_amplitude}", n_macroparticles, impulse_duration, intensity, perturbation_amplitude).lazy_run()
    plots(s, *args, frames="few")
# s = laser("production_run_nolaser_e-3", n_macroparticles, impulse_duration, 0, perturbation_amplitude).lazy_run()
# plots(s, *args, frames="few")
# s = laser("field_only", 0, impulse_duration, 1e21, perturbation_amplitude).lazy_run()
# plots(s, *args, frames="few")
