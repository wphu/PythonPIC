# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.helper_functions.physics import did_it_thermalize
from pythonpic.configs.run_twostream import two_stream_instability, plots


args = plotting_parser("Two stream instability")
S = two_stream_instability("TS1",
                       NG=512,
                       N_electrons=4096,
                       plasma_frequency=0.05 / 4,
                       ).lazy_run()
print(did_it_thermalize(S))
plots(S, *args)
S = two_stream_instability("TS2",
                       NG=512,
                       N_electrons=4096,
                       plasma_frequency=0.05,
                       ).lazy_run()
print(did_it_thermalize(S))
plots(S, *args)
S = two_stream_instability("TS3",
                       NG=512,
                       N_electrons=4096,
                       plasma_frequency=0.05 * 10,
                       ).lazy_run()
print(did_it_thermalize(S))
plots(S, *args)
S = two_stream_instability("TSRANDOM1",
                       NG=512,
                       N_electrons=4096,
                       vrandom=1e-1,
                       ).lazy_run()
print(did_it_thermalize(S))
plots(S, *args)
S = two_stream_instability("TSRANDOM2",
                       NG=512, N_electrons=4096,
                       vrandom=1e-1).lazy_run()
print(did_it_thermalize(S))
plots(S, *args)