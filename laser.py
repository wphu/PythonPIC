# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.run_laser import laser, impulse_duration, n_macroparticles, plots

args = plotting_parser("Hydrogen shield")
s =laser("field_only", 0, impulse_duration).lazy_run()
plots(s, *args)
s =laser("few_particles", 10000, impulse_duration).lazy_run()
plots(s, *args)
s = laser("few_particles_short_pulse", 10000, impulse_duration/10).lazy_run()
plots(s, *args)
s = laser("production_run_short_pulse", n_macroparticles, impulse_duration/10).lazy_run()
plots(s, *args)
s = laser("production_run", n_macroparticles, impulse_duration).lazy_run()
plots(s, *args)
