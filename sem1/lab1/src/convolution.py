from typing import List


def conv_time(x: List[complex], y: List[complex]) -> List[complex]:
    """
    Свёртка по определению во временной области.
    Длины: len(x) = N, len(y) = M, результат длины N+M-1.
    """
    N = len(x)
    M = len(y)
    L = N + M - 1

    x_extended = [0j] * L
    for n in range(N):
        x_extended[n + M - 1] = x[n]

    y_reverse = [y [M - 1 - n] for n in range(M)]
    y_extended = [0j] * L
    for n in range(M):
        y_extended[n] = y_reverse[n]

    s_convolved = [0j] * L
    for n in range(L):
        acc = 0j
        for k in range(L):
            acc += x_extended[k] * y_extended[k]
        s_convolved[n] = acc

        for k in range(L-1,0,-1):
            y_extended[k] = y_extended[k-1]
        y_extended[0] = 0j
    return s_convolved


def conv_fft(x: List[complex], y: List[complex]) -> List[complex]:
    """Свёртка с использованием БПФ (БПФ_В или numpy.fft)."""
    raise NotImplementedError
