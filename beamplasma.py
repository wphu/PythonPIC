# coding=utf-8
import numpy as np

from pythonpic import weakbeam_instability, plotting_parser, plots


args = plotting_parser("Weak beam instability")
np.random.seed(0)
s = weakbeam_instability("beamplasma1").lazy_run()
plots(s, *args, alpha=0.5)