# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.run_laser import laser, impulse_duration, n_macroparticles, plots

args = plotting_parser("Hydrogen shield")
number_particles = 1000
s = laser(f"{number_particles}_stability", number_particles, impulse_duration,
          0, 0, runtime_multiplier=0.5).test_run()
if any(args):
    plots(s, *args, frames="few")
