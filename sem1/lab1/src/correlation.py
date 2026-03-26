from typing import List
from fft_time_decimation import fft_dit, ifft_dit

def corr_time(x: List[complex], y: List[complex]) -> List[complex]:
    """Корреляция по определению во временной области."""
    N = len(x)
    assert len(y) == N

    m_values = list(range(-(N - 1), N))
    correlation = [0j] * len(m_values)

    for idx, m in enumerate(m_values):
        acc = 0j
        for n in range(N):
            j = n - m
            if 0 <= j < N:  # n + m принадлежит [0, N-1]
                acc += x[n] * y[j]
        correlation[idx] = acc

    return correlation


def corr_fft(x: List[complex], y: List[complex]) -> List[complex]:
    """Корреляция через БПФ по теореме корреляции."""
    N = len(x)
    assert len(y) == N

    # длина БПФ
    N_conv = 2 * N - 1
    N_fft = 1
    while N_fft < N_conv:
        N_fft *= 2

    # дописать нули до конца x
    x_extended = [0j] * N_fft
    for n in range(N):
        x_extended[n] = x[n]
    # дописать нули до конца y
    y_extended = [0j] * N_fft
    for n in range(N):
        y_extended[n] = y[n]

    # FFT для обеих функций
    X_fft = fft_dit(x_extended)
    Y_fft = fft_dit(y_extended)

    # поэлементное умножение спектров
    Z_fft = [0j] * N_fft
    for n in range(N_fft):
        Z_fft[n] = X_fft[n] * Y_fft[n].conjugate() # комплексное сопряжение

    # IFFT для z
    correlation_cyclic = ifft_dit(Z_fft)
    # нулевой сдвиг должен встать посередине
    correlation_cyclic = correlation_cyclic[-(N-1):] + correlation_cyclic[:N]
    # теперь берём ровно 2N-1 отсчётов
    correlation_linear = correlation_cyclic[:N_conv]

    return correlation_linear
