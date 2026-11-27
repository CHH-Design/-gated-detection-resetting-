"""Verify the Erlang-stage threshold example. Run with Python 3 and NumPy."""
import math
from decimal import Decimal, localcontext

import numpy as np


def direct_cycle_decimal(kappa, reset, forward, backward, stages, retention):
    """Sum full cycle durations at 60 digits before subtracting the baseline."""
    def inverse(matrix):
        block = np.concatenate((matrix.copy(), identity.copy()), axis=1)
        for j in range(3):
            pivot = j + max(range(3-j), key=lambda i: abs(block[j+i, j]))
            block[[j, pivot]] = block[[pivot, j]]
            block[j] = block[j] / block[j, j]
            for i in range(3):
                if i != j:
                    block[i] = block[i] - block[i, j] * block[j]
        return block[:, 3:]

    with localcontext() as context:
        context.prec = 60
        k, r, f, b, c = map(lambda x: Decimal(str(x)),
                            (kappa, reset, forward, backward, retention))
        identity = np.vectorize(Decimal, otypes=[object])(np.eye(3, dtype=int).astype(str))
        e0 = identity[:, 0]
        generator = (f * np.roll(identity, 1, axis=1) + b * np.roll(identity, -1, axis=1)
                     - (f + b) * identity)
        reach = np.linalg.matrix_power(k * inverse((k+r) * identity - generator), stages)
        failed = (identity - np.outer(reach @ e0, e0)) @ (r * inverse(r * identity - generator))
        restart = c * identity + (1-c) * np.full((3, 3), Decimal(1)/3)
        duration = (1-reach @ e0) / r
        values = inverse(identity - failed @ restart) @ duration
        baseline = (3 * ((k+r)/k)**stages - 1) / r
        return float(sum(values)/3 - baseline)


def main():
    a, b, reset = 1.0, 0.57, 0.1
    stages, theta = 9, 4 * math.pi / 3
    kappa = b / math.tan(theta / stages) - a - reset
    forward, backward = a / 3 + b / math.sqrt(3), a / 3 - b / math.sqrt(3)
    alpha, gamma = math.atan(b / (a + reset)), math.atan(b / a)
    target = 1.5 * math.pi
    optimum = (a + reset - b / math.tan((target - theta + alpha) / 2)) / reset
    pass_probability = (kappa / (kappa + reset)) ** stages
    base = (3 / pass_probability - 1) / reset
    mode = ((kappa + reset) / (kappa + reset + a - 1j * b)) ** stages
    assert 0 < backward < forward and kappa > 0
    assert (8 - 1) * alpha + 2 * gamma < target < (9 - 1) * alpha + 2 * gamma
    assert theta + alpha < target < theta - alpha + 2 * gamma
    print(f'k+={forward:.12g} k-={backward:.12g} kappa={kappa:.12g} '
          f'P(S<R)={pass_probability:.12g} c*={optimum:.12g}')
    print(f'phase endpoints={theta + alpha:.12g}, {theta-alpha+2*gamma:.12g}; '
          f'T(0)={base:.12g}')

    identity = np.eye(3)
    generator = (forward * np.roll(identity, 1, axis=1)
                 + backward * np.roll(identity, -1, axis=1)
                 - (forward + backward) * identity)
    e0 = identity[:, 0]
    # Direct cycle recursion: each Erlang stage contributes one resolvent.
    hit_stage = np.linalg.solve((kappa + reset) * identity - generator, kappa * identity)
    reach = np.linalg.matrix_power(hit_stage, stages)
    at_reset = np.linalg.solve(reset * identity - generator, reset * identity)
    failed = (identity - np.outer(reach @ e0, e0)) @ at_reset
    duration = (1 - reach @ e0) / reset

    # Remove the constant mode before taking powers, so the tiny excess
    # can be checked without subtracting two nearly equal total times.
    basis = np.array([[1, 1], [-1, 1], [0, -2]]) / np.sqrt([2, 6])
    reduced = basis.T @ generator @ basis
    unit = np.eye(2)
    conditional_stage = np.linalg.solve((kappa + reset) * unit - reduced,
                                        (kappa + reset) * unit)
    crossing = np.linalg.matrix_power(conditional_stage, stages)
    open_coordinates = basis.T @ e0
    for c in (0.0, optimum, 1.0):
        d = a + reset * (1 - c)
        excess = 2 * c * (mode / (d - 1j * b)).real
        matrix_excess = 3 * c * open_coordinates @ np.linalg.solve(
            reset * (1 - c) * unit - reduced, crossing @ open_coordinates)
        np.testing.assert_allclose(matrix_excess, excess, rtol=2e-12, atol=1e-29)
        memory = c * identity + (1-c) * np.ones((3, 3)) / 3
        matrix_total = np.linalg.solve(identity - failed @ memory, duration).mean()
        np.testing.assert_allclose(matrix_total, base + excess, rtol=2e-9)
        direct_excess = direct_cycle_decimal(kappa, reset, forward, backward, stages, c)
        np.testing.assert_allclose(direct_excess, excess, rtol=2e-12, atol=1e-29)
        optimum_d = a + reset * (1 - optimum)
        gap = (2 * abs(mode) * math.hypot(a + reset, b) * b * reset * (c-optimum)**2
               / ((optimum_d**2 + b**2) * (d**2 + b**2)))
        optimum_excess = 2 * optimum * (mode / (optimum_d - 1j * b)).real
        np.testing.assert_allclose(excess - optimum_excess, gap, rtol=1e-11, atol=1e-29)
        print(f'c={c:.12g} total={base+excess:.12g} '
              f'T(c)-T(0)={excess:.12g} T(c)-T(c*)={gap:.12g}')


if __name__ == '__main__':
    main()
