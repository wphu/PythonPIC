from pythonpic import plotting_parser, plots, laser_shield, plots
from pythonpic.configs.run_laser import impulse_duration, n_macroparticles

args = plotting_parser("Hydrogen shield")
s =laser_shield("field_only", 0, impulse_duration).lazy_run()
plots(s, *args, frame_to_draw="anim_snapshots")
s =laser_shield("few_particles", 10000, impulse_duration).lazy_run()
plots(s, *args, frame_to_draw="anim_snapshots")
s = laser_shield("few_particles_short_pulse", 10000, impulse_duration/10).lazy_run()
plots(s, *args, frame_to_draw="anim_snapshots")
s = laser_shield("production_run_short_pulse", n_macroparticles, impulse_duration/10).lazy_run()
plots(s, *args, frame_to_draw="anim_snapshots")
# s = laser_shield("production_run", n_macroparticles, impulse_duration).lazy_run()
# plots(s, *args, frame_to_draw="anim_snapshots")
