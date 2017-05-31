from pythonpic import wave_propagation, plotting_parser, plots, BoundaryCondition


args = plotting_parser("Wave propagation")
for filename, boundary_function in zip(["Wave", "Envelope", "Laser"],
                                       [BoundaryCondition.Laser(1, 1, 10, 3).laser_wave,
                                        BoundaryCondition.Laser(1, 1, 10, 3).laser_envelope,
                                        BoundaryCondition.Laser(1, 1, 10, 3).laser_pulse,
                                        ]):
    s = wave_propagation(filename, bc=boundary_function).lazy_run()
    plots(s, *args, alpha=0.5)