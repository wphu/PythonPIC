# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.run_laser import laser, impulse_duration, n_macroparticles, plots

args = plotting_parser("Hydrogen shield")
perturbation_amplitude = 0
s = laser("production_run_nolaser_e-3", n_macroparticles, impulse_duration, 0, perturbation_amplitude).lazy_run()
plots(s, *args, frames="few")
s = laser("field_only", 0, impulse_duration, 1e21, perturbation_amplitude).lazy_run()
plots(s, *args, frames="few")
s = laser("production_run_23_e-3", n_macroparticles, impulse_duration, 1e23, perturbation_amplitude).lazy_run()
plots(s, *args, frames="few")
s = laser("production_run_22_e-3", n_macroparticles, impulse_duration, 1e22, perturbation_amplitude).lazy_run()
plots(s, *args, frames="few")
s = laser("production_run_21_e-3", n_macroparticles, impulse_duration, 1e21, perturbation_amplitude).lazy_run()
plots(s, *args, frames="few")
# s = laser("production_run_22_01", n_macroparticles, impulse_duration, 1e22, 0.01).lazy_run()
# plots(s, *args)
# s =laser("few_particles", 10000, impulse_duration).lazy_run()
# plots(s, *args)
# s = laser("few_particles_short_pulse", 10000, impulse_duration/10).lazy_run()
# plots(s, *args)
# s = laser("production_run_short_pulse", n_macroparticles, impulse_duration/10).lazy_run()
# plots(s, *args)
