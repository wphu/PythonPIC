# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.dense_uniform import uniform, plots

args = plotting_parser("stab2")
number_particles = 10000
for number_particles in [10000, 50000]:
    s = uniform(f"{number_particles}_stability2", number_particles).test_run()
    if any(args):
        plots(s, *args, frames="few")
    del s
