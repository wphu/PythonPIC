# Algorithms for deposition of charge and currents on grid
# coding=utf-8
import numpy as np


def charge_density_deposition(x, dx, x_particles, particle_charge):
    """scatters charge from particles to grid
    uses linear interpolation
    x_i | __________p___| x_i+1
    for a particle $p$ in cell $i$ of width $dx$ the location in cell is defined as
    $$X_p = x_p - x_i$$
    then, $F_r = X_p/dx$ is the fraction of charge going to the right side of the cell
    (as $X_p \to dx$, the particle is closer to the right cell)
    while $F_l = 1-F_r$ is the fraction of charge going to the left

    numpy.bincount is used to count particles in cells
    the weights are the fractions for each cell

    to change the index for the right going  (keeping periodic boundary conditions)
    numpy.roll is used
    """
    logical_coordinates = (x_particles / dx).astype(int)
    right_fractions = x_particles / dx - logical_coordinates
    left_fractions = 1 - right_fractions
    charge_to_right = particle_charge * right_fractions
    charge_to_left = particle_charge * left_fractions
    charge_hist_to_right = np.bincount(logical_coordinates+1, charge_to_right, minlength=x.size+1)
    charge_hist_to_left = np.bincount(logical_coordinates, charge_to_left, minlength=x.size+1)
    return charge_hist_to_right + charge_hist_to_left

def longitudinal_current_deposition(j_x, x_velocity, x_particles, dx, dt, q):
    """
    
    Parameters
    ----------
    j_x : ndarray
        current in x direction
    x_velocity : ndarray
        x velocity
    x_particles : ndarray
        x position
    dx : float
        grid cell size
    dt : float
        timestep
    q : float
        charge to deposit. for `Species`, this is `eff_q`

    Returns
    -------

    """
    print("Longitudinal current deposition")
    active = np.ones_like(x_particles, dtype=bool)
    time = np.ones_like(x_particles) * dt
    epsilon = dx * 1e-9
    counter = 0
    actives = []
    active = x_velocity != 0
    x_particles = x_particles[active]
    x_velocity = x_velocity[active]
    time = time[active]

    while active.any():
        counter += 1
        actives.append(active.sum())
        if counter > 4:
            # import matplotlib.pyplot as plt
            # plt.plot(actives)
            # plt.show()
            raise Exception("Infinite recurrence!")

        # print(j_x, x_velocity, x_particles, time, dx, dt, q)
        # noinspection PyUnresolvedReferences
        logical_coordinates_n = (x_particles / dx).astype(int)
        # logical_coordinates_n2 = (x_particles // dx).astype(int)
        # results = [logical_coordinates_n, logical_coordinates_n2]
        # labels = ["/", "//", "floor_divide"]
        # def plot():
        #     import matplotlib.pyplot as plt
        #     for r, lab in zip(results, labels):
        #         plt.plot(r,  label=lab, alpha=0.5)
        #     plt.legend()
        #     plt.show()
        # for r1 in results:
        #     for r2 in results:
        #         if r2 is not r1:
        #             assert np.allclose(r1, r2), plot()
        #             # assert (r1 == r2).all()

        particle_in_left_half = x_particles / dx - logical_coordinates_n <= 0.5
        # TODO: what happens when particle is at center
        particle_in_right_half = x_particles / dx - logical_coordinates_n > 0.5
        velocity_to_left = x_velocity < 0
        velocity_to_right = x_velocity > 0

        case1 = particle_in_left_half & velocity_to_left
        case2 = particle_in_right_half & velocity_to_left
        case3 = particle_in_right_half & velocity_to_right
        case4 = particle_in_left_half & velocity_to_right

        t1 = np.empty_like(x_particles)
        # case1
        t1[case1] = -(x_particles[case1] - (logical_coordinates_n[case1]) * dx) / x_velocity[case1]
        t1[case2] = -(x_particles[case2] - (logical_coordinates_n[case2] + 0.5) * dx) / x_velocity[case2]
        t1[case3] = ((logical_coordinates_n[case3] + 1) * dx - x_particles[case3]) / x_velocity[case3]
        t1[case4] = ((logical_coordinates_n[case4] + 1.5) * dx - x_particles[case4]) / x_velocity[case4]
        switches_cells = t1 < time

        logical_coordinates_depo = logical_coordinates_n.copy()
        logical_coordinates_depo[case2 | case3] += 1
        if (~switches_cells).any():
            nonswitching_current_contribution = q * x_velocity[~switches_cells] * time[~switches_cells]
            j_x += np.bincount(logical_coordinates_depo[~switches_cells] + 1, nonswitching_current_contribution,
                               minlength=j_x.size)

        new_time = time - t1
        if switches_cells.any():
            switching_current_contribution = q * x_velocity[switches_cells] * t1[switches_cells] / dt
            j_x += np.bincount(logical_coordinates_depo[switches_cells] + 1, switching_current_contribution, minlength=j_x.size)

        new_locations = np.empty_like(x_particles)
        new_locations[case1] = (logical_coordinates_n[case1]) * dx - epsilon
        new_locations[case2] = (logical_coordinates_n[case2] + 0.5) * dx - epsilon
        new_locations[case3] = (logical_coordinates_n[case3] + 1) * dx + epsilon
        new_locations[case4] = (logical_coordinates_n[case4] + 0.5) * dx + epsilon
        if counter > 2:
            string =f"""iteration:\t{counter}
                  dx: {dx}
                  number of actives: {active.sum()}
                  fraction of case1 (in left, to left): {case1.sum() / active.sum()}
                  fraction of case2 (in right, to left): {case2.sum() / active.sum()}
                  fraction of case3 (in right, to right): {case3.sum() / active.sum()}
                  fraction of case4 (in left, to right): {case4.sum() / active.sum()}
                  x_particles: {x_particles},
                  indices: {logical_coordinates_n},
                  new locations: {new_locations},
                  t1 {t1},
                  distance to cover in grid cell units: {x_velocity*new_time/dx},
                  time left in dt units: {new_time/dt},
                  distance to cover: {new_time*x_velocity},
                  distance between current and next: {x_particles - new_locations},
                  \n\n"""
            modified_string = "\n".join(line.strip() for line in string.splitlines())
            print(modified_string)

        active = switches_cells
        x_particles = new_locations[active]
        x_velocity = x_velocity[active]
        time = new_time[active]
        active = np.ones_like(x_particles, dtype=bool)
    print("finished logitudinal")


def transversal_current_deposition(j_yz, velocity, x_particles, dx, dt, q):
    # TODO: optimize this algorithm

    print("Transversal deposition")
    epsilon = 1e-10 * dx
    time = np.ones_like(x_particles) * dt
    active = np.ones_like(x_particles, dtype=bool)
    counter = 0
    actives = []
    while active.any():
        counter += 1
        if counter > 4:
            # import matplotlib.pyplot as plt
            # plt.plot(actives)
            # plt.show()
            raise Exception("Infinite recurrence!")
        print(active.sum())
        velocity = velocity[active]
        x_velocity = velocity[:, 0]
        y_velocity = velocity[:, 1]
        z_velocity = velocity[:, 2]
        x_particles = x_particles[active]
        time = time[active]


        logical_coordinates_n = (x_particles / dx).astype(int)
        particle_in_left_half = x_particles / dx - logical_coordinates_n < 0.5
        particle_in_right_half = ~particle_in_left_half

        velocity_to_left = x_velocity < 0
        velocity_to_right = ~velocity_to_left

        t1 = np.empty_like(x_particles)
        s = np.empty_like(x_particles)

        case1 = particle_in_left_half & velocity_to_left
        case2 = particle_in_left_half & velocity_to_right
        case3 = particle_in_right_half & velocity_to_right
        case4 = particle_in_right_half & velocity_to_left
        t1[case1] = - (x_particles[case1] - logical_coordinates_n[case1] * dx) / x_velocity[case1]
        s[case1] = logical_coordinates_n[case1] * dx - epsilon
        t1[case2] = ((logical_coordinates_n[case2] + 0.5) * dx - x_particles[case2]) / x_velocity[case2]
        s[case2] = (logical_coordinates_n[case2] + 0.5) * dx + epsilon
        t1[case3] = ((logical_coordinates_n[case3] + 1) * dx - x_particles[case3]) / x_velocity[case3]
        s[case3] = (logical_coordinates_n[case3] + 1) * dx + epsilon
        t1[case4] = -(x_particles[case4] - (logical_coordinates_n[case4] + 0.5) * dx) / x_velocity[case4]
        s[case4] = (logical_coordinates_n[case4] + 0.5) * dx - epsilon

        time_overflow = time - t1
        switches_cells = time_overflow > 0
        time_in_this_iteration = time.copy()
        time_in_this_iteration[switches_cells] = t1[switches_cells]

        jy_contribution = q * y_velocity / dt * time_in_this_iteration
        jz_contribution = q * z_velocity / dt * time_in_this_iteration

        # noinspection PyUnresolvedReferences
        sign = particle_in_left_half.astype(int) * 2 - 1
        distance_to_current_cell_center = (logical_coordinates_n + 0.5) * dx - x_particles
        s0 = (1 - sign * distance_to_current_cell_center / dx)
        change_in_coverage = sign * x_velocity * time_in_this_iteration / dx
        s1 = s0 + change_in_coverage
        w = 0.5 * (s0 + s1)

        logical_coordinates_depo = logical_coordinates_n.copy()
        logical_coordinates_depo[particle_in_left_half] -= 1
        logical_coordinates_depo[particle_in_right_half] += 1

        y_contribution_to_current_cell = w * jy_contribution
        y_contribution_to_next_cell = (1 - w) * jy_contribution
        z_contribution_to_current_cell = w * jz_contribution
        z_contribution_to_next_cell = (1 - w) * jz_contribution

        # assert (w <= 1).all(), f"w {w} > 1"
        # assert (w >= 0.5).all(), f"w {w} < 0.5"
        # assert (1-w <= 0.5).all(), f"1-w {1-w} > 0.5"
        # assert (1-w >= 0).all(), f"1-w {1-w} < 0"
        # if (~switches_cells).all():
        #     print(f"x0: {x_particles}\t v: {velocity}\t x1: {s} \t time: {time_in_this_iteration}")
        #     print("====J contribution=====")
        #     print(logical_coordinates_depo)
        #     print("y", y_contribution_to_current_cell)
        #     print("z", z_contribution_to_current_cell)
        #     print("to current cell:", logical_coordinates_n)
        #     print("y", y_contribution_to_next_cell)
        #     print("z", y_contribution_to_next_cell)
        #     print("to cell:", logical_coordinates_depo)

        j_yz[:, 0] += np.bincount(logical_coordinates_n + 2, y_contribution_to_current_cell, minlength=j_yz[:, 1].size)
        j_yz[:, 1] += np.bincount(logical_coordinates_n + 2, z_contribution_to_current_cell, minlength=j_yz[:, 1].size)

        j_yz[:, 0] += np.bincount(logical_coordinates_depo + 2, y_contribution_to_next_cell, minlength=j_yz[:, 1].size)
        j_yz[:, 1] += np.bincount(logical_coordinates_depo + 2, z_contribution_to_next_cell, minlength=j_yz[:, 1].size)

        if counter > 2:
            print(counter,
                  dx,
                  active.sum(),
                  case1.sum() / active.sum(),
                  x_particles, # the first array in there
                  logical_coordinates_n,
                  s,
                  # logical_coordinates_n - x_particles/dx,
                  # (x_particles - logical_coordinates_n * dx)/dx,
                  # (new_locations - logical_coordinates_n * dx)/dx,
                  t1,
                  x_velocity*time_overflow/dx,
                  time_overflow/dt,
                  time_overflow*x_velocity,
                  x_particles - time_overflow,
                  "\n\n"
                  )
        active = switches_cells
        time = time_overflow
        x_particles = s

# def current_density_deposition(x, dx, x_particles, particle_charge, velocity):
#     """scatters charge from particles to grid
#     uses linear interpolation
#     x_i | __________p___| x_i+1
#     for a particle $p$ in cell $i$ of width $dx$ the location in cell is defined as
#     $$X_p = x_p - x_i$$
#     then, $F_r = X_p/dx$ is the fraction of charge going to the right side of the cell
#     (as $X_p \to dx$, the particle is closer to the right cell)
#     while $F_l = 1-F_r$ is the fraction of charge going to the left
#
#     numpy.bincount is used to count particles in cells
#     the weights are the fractions for each cell
#
#     to change the index for the right going  (keeping periodic boundary conditions)
#     numpy.roll is used
#     """
#     logical_coordinates = (x_particles / dx).astype(int)
#     right_fractions = x_particles / dx - logical_coordinates
#     left_fractions = -right_fractions + 1.0
#     current_to_right = particle_charge * velocity * right_fractions.reshape(x_particles.size, 1)
#     current_to_left = particle_charge * velocity * left_fractions.reshape(x_particles.size, 1)
#     # OPTIMIZE: use numba with this
#     current_hist = np.zeros((x.size, 3))
#     for dim in range(3):
#         current_hist[:, dim] += np.bincount(logical_coordinates, current_to_left[:, dim], minlength=x.size)
#         current_hist[:, dim] += np.roll(np.bincount(logical_coordinates, current_to_right[:, dim], minlength=x.size),
#                                         +1)
#     return current_hist


def interpolateField(x_particles, scalar_field, x, dx):
    """gathers field from grid to particles

    the reverse of the algorithm from charge_density_deposition

    there is no need to use numpy.bincount as the map is
    not N (number of particles) to M (grid), but M to N, N >> M
    """
    logical_coordinates = (x_particles / dx).astype(int)
    NG = scalar_field.size
    right_fractions = x_particles / dx - logical_coordinates
    field = (1 - right_fractions) * scalar_field[logical_coordinates] + \
            right_fractions * scalar_field[(logical_coordinates + 1) % NG]
    return field
