"""Reproduce the single-encounter gate-memory checks. Python 3 + NumPy.

Run: python reproduce.py
Each simulation samples exact continuous reset times and exact ring increments.
The CSV next to this script reports unscaled times; c is the retention probability.
"""
from pathlib import Path
import csv
import math
import numpy as np


def spectral_time(m, forward, backward, tau, rate, c):
    angles = 2 * np.pi * np.arange(1, m) / m
    modes = forward * np.expm1(1j * angles) + backward * np.expm1(-1j * angles)
    memory = np.sum((np.exp(modes * tau) / (rate * (1 - c) - modes)).real)
    return (m * np.exp(rate * tau) - 1) / rate + c * memory


def matrix_time(m, forward, backward, tau, rate, c):
    """Solve the reset-cycle recursion using a Poisson uniformization series."""
    identity = np.eye(m)
    jump_rate = forward + backward
    step = (forward * np.roll(identity, 1, axis=1)
            + backward * np.roll(identity, -1, axis=1)) / jump_rate
    generator = jump_rate * (step - identity)
    poisson_mean = tau * jump_rate
    weight = np.exp(-poisson_mean)
    power = identity
    propagate = weight * identity
    n = 0
    while True:
        n += 1
        weight *= poisson_mean / n
        power = power @ step
        propagate += weight * power
        if n > poisson_mean and weight / (1 - poisson_mean / (n + 1)) < 1e-15:
            break
    survive_to_cross = np.exp(-rate * tau)
    reset_propagator = rate * np.linalg.solve(rate * identity - generator, identity)
    early_reset = (identity - survive_to_cross * propagate) @ reset_propagator
    failed_cross = survive_to_cross * propagate.copy()
    failed_cross[:, 0] = 0
    redraw = c * identity + (1 - c) * np.ones((m, m)) / m
    next_cycle = (early_reset + failed_cross @ reset_propagator) @ redraw
    mean_duration = (1 - survive_to_cross + failed_cross.sum(axis=1)) / rate
    return np.linalg.solve(identity - next_cycle, mean_duration).mean()


def monte_carlo(m, forward, backward, tau, rate, c, count, seed):
    rng = np.random.default_rng(seed)
    totals = np.zeros(count)
    active = np.arange(count)
    gate = rng.integers(m, size=count)

    def advance(state, duration):
        ahead = rng.poisson(forward * duration)
        back = rng.poisson(backward * duration)
        return (state + ahead - back) % m

    while active.size:
        reset_time = rng.exponential(1 / rate, size=active.size)
        first_leg = np.minimum(reset_time, tau)
        totals[active] += first_leg
        gate = advance(gate, first_leg)
        crossed = reset_time >= tau
        keep = ~(crossed & (gate == 0))
        active, gate = active[keep], gate[keep]
        past = crossed[keep]
        after_crossing = reset_time[keep][past] - tau
        totals[active[past]] += after_crossing
        gate[past] = advance(gate[past], after_crossing)
        redraw = rng.random(active.size) >= c
        gate[redraw] = rng.integers(m, size=np.count_nonzero(redraw))
    mean = totals.mean()
    se = totals.std(ddof=1) / math.sqrt(count)
    return mean, se


def main():
    m, tau, rate, count = 12, 4.0, 0.2, 150_000
    columns = ['gate', 'm', 'forward', 'backward', 'tau', 'reset_rate', 'retention',
               'trajectories', 'seed', 'spectral_time', 'matrix_time', 'mc_time',
               'mc_se', 'ci95_low', 'ci95_high']
    with Path(__file__).with_name('checks.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for i, (label, forward, backward) in enumerate(
                [('driven', 1.0, 0.05), ('reversible', 0.525, 0.525)]):
            for j, c in enumerate((0., 0.5, 1.)):
                seed = 20261127 + 10 * i + j
                params = m, forward, backward, tau, rate, c
                spectral = spectral_time(*params)
                matrix = matrix_time(*params)
                np.testing.assert_allclose(matrix, spectral, rtol=1e-12)
                mean, se = monte_carlo(*params, count, seed)
                low, high = mean - 1.959963984540054 * se, mean + 1.959963984540054 * se
                writer.writerow([label, *params, count, seed, spectral, matrix, mean, se, low, high])
                print(f'{label:10} c={c:.1f} exact={spectral:.6f} '
                      f'MC={mean:.6f} CI=({low:.6f}, {high:.6f})')


if __name__ == '__main__':
    main()
