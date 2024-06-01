import numpy as np

from avp.audio.signal import ndsignal


def sinusoid(hz: int, sr: int = 100, seconds: float = 1.0):
    t = np.arange(0, seconds, 1/sr)

    return ndsignal(samples=np.sin(2 * np.pi * hz * t), sr=sr)


def noise(factor: float = 0.5, samples: int = 100, seed: int = 100):
    np.random.seed(seed)

    return factor * np.random.normal(100)


def multisinoid(F: list, sr=100, seconds: float = 1.0):
    all = list()

    for f in F:
        all.append(sinusoid(f, sr, seconds))

    return np.sum(all)


def randsignal(max_hz, min_hz, max_a, min_a, **noise_params):
    pass