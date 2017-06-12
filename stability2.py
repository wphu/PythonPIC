# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.dense_uniform import uniform, plots, number_cells

args = plotting_parser("stab2")
for number_particles, n_cells in [
    # [10000, int(number_cells/2)], #stable
    [10000, int(number_cells/3)], # kinda stable but weird
    # [20000, number_cells], # stable
    [20000, int(number_cells/2)], # CHECK
    ]:
    s = uniform(f"{number_particles}_{n_cells}_stability2", number_particles, n_cells).lazy_run()
    if any(args):
        plots(s, *args, frames="few")
    del s
