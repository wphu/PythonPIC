# coding=utf-8
import numpy as np

from pythonpic import plotting_parser
from pythonpic.configs.run_beamplasma import weakbeam_instability, plots


args = plotting_parser("Weak beam instability")
np.random.seed(0)
s = weakbeam_instability("beamplasma1").lazy_run()
plots(s, *args, alpha=0.5)