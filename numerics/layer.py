"""Large driven rings: deterministic Fourier means, a dense renewal check, and limits."""
import numpy as np
from numpy.polynomial.legendre import leggauss


def excess_per_state(m, c, forward, backward, tau, reset):
    angle = 2 * np.pi * np.arange(1, m) / m
    modes = (-2 * (forward + backward) * np.sin(angle / 2)**2
             + 1j * (forward - backward) * np.sin(angle))
    return c * np.real(np.exp(tau * modes) / (reset * (1-c) - modes)).sum() / m


def dense_mean(m, c, forward, backward, tau, reset):
    eye = np.eye(m)
    generator = (forward * np.roll(eye, 1, axis=1)
                 + backward * np.roll(eye, -1, axis=1) - (forward+backward)*eye)
    values, vectors = np.linalg.eig(generator)
    propagator = np.real((vectors * np.exp(tau*values)) @ np.linalg.inv(vectors))
    hit = np.exp(-reset*tau) * propagator[:, 0]
    reset_epoch = np.linalg.solve(reset*eye-generator, reset*eye)
    failure = reset_epoch - np.outer(hit, reset_epoch[0])
    retained = c*eye + (1-c)/m
    return np.linalg.solve(eye - failure @ retained, (1-hit)/reset).mean()


def layer_limit(z, forward, backward, tau):
    drift = forward - backward
    points, weights = leggauss(64)
    times = tau * (1 + points) / 2
    p_line = (np.exp(-(forward + backward) * times)
              * np.i0(2 * np.sqrt(forward * backward) * times))
    occupation = tau/2 * np.dot(weights, p_line)
    initial = 1/(2*drift) if z == 0 else -1/(drift*np.expm1(-z/drift)) - 1/z
    return initial - occupation


def main():
    forward, backward, tau, reset = 1., .05, 2., .5
    m = 12
    for c in [0., .5, 1.]:
        total = (m * np.exp(reset*tau)-1)/reset + m*excess_per_state(
            m, c, forward, backward, tau, reset)
        dense = dense_mean(m, c, forward, backward, tau, reset)
        np.testing.assert_allclose(total, dense, rtol=1e-12)
        print(f"m={m} c={c:.2f}: Fourier={total:.10f} dense={dense:.10f}")
    print("z, m=64, m=256, m=1024, limit")
    for z in [0., 2., 8.]:
        finite = [excess_per_state(m, 1-z/(m*reset), forward, backward, tau, reset)
                  for m in [64, 256, 1024]]
        limit = layer_limit(z, forward, backward, tau)
        assert abs(finite[-1] - limit) < 0.003
        print(f"{z:g}, " + ", ".join(f"{x:.7f}" for x in [*finite, limit]))


if __name__ == "__main__":
    main()
