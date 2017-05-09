import run_beamplasma
import run_coldplasma
import run_twostream
import run_wave

for conf in run_wave, run_coldplasma, run_twostream, run_beamplasma:
    conf.main()
