# coding=utf-8
from pythonpic import plotting_parser
from pythonpic.configs.run_laser import laser, impulse_duration, n_macroparticles, plots

def run():
    perturbation_amplitude = 0
    number_particles = 10000
    power = 21
    intensity = 10**power
    s = laser(f"{number_particles}_run_{power}_{perturbation_amplitude}", number_particles, impulse_duration, intensity, perturbation_amplitude).run()
    return s

if __name__ == "__main__":
    s = run()
    args = plotting_parser("Hydrogen shield")
    if any(args):
        plots(s, *args, frames="few")
