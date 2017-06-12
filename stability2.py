# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.dense_uniform import uniform, plots

args = plotting_parser("stab2")
number_particles = 10000
s = uniform(f"{number_particles}_stability2", number_particles).test_run()
if any(args):
    plots(s, *args, frames="few")
