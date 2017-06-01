# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.run_coldplasma import plots, cold_plasma_oscillations


args = plotting_parser("Cold plasma oscillations")
plasma_frequency = 1
push_mode = 2
N_electrons = 1024
NG = 64
qmratio = -1

S = cold_plasma_oscillations(f"CO1", qmratio=qmratio, plasma_frequency=plasma_frequency, NG=NG,
                             N_electrons=N_electrons, push_mode=push_mode, save_data=False).lazy_run()
plots(S, *args)
