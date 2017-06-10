# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.run_laser import laser, impulse_duration, n_macroparticles, plots

args = plotting_parser("Hydrogen shield")
perturbation_amplitude = 0
powers = range(23, 22, -1)
for power in powers:
    intensity = 10**power
    s = laser(f"field_only_{power}", 0, impulse_duration, intensity, perturbation_amplitude=0).lazy_run()
    plots(s, *args, frames="few")
    del s

    s = laser(f"production_run_{power}_{perturbation_amplitude}",
              n_macroparticles,
              impulse_duration,
              intensity,
              perturbation_amplitude=0).lazy_run()
    if any(args):
        plots(s, *args, frames="few")
    del s
# s = laser("production_run_nolaser_e-3", n_macroparticles, impulse_duration, 0, perturbation_amplitude).lazy_run()
# plots(s, *args, frames="few")
