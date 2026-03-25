from typing import List
from fft_time_decimation import fft_dit, ifft_dit

def conv_time(x: List[complex], y: List[complex]) -> List[complex]:
    """
    Свёртка по определению во временной области.
    Длины: len(x) = N, len(y) = M, результат длины N+M-1.
    """
    N = len(x)
    M = len(y)
    L = N + M - 1

    # x дополнить нулями слева
    x_extended = [0j] * L
    for n in range(N):
        x_extended[n + M - 1] = x[n]

    # y перевернуть по оси X
    y_reverse = [y [M - 1 - n] for n in range(M)]
    # y дополнить нулями справа
    y_extended = [0j] * L
    for n in range(M):
        y_extended[n] = y_reverse[n]

    # размер свертки
    s_convolved = [0j] * L
    for n in range(L):
        window = 0j # окно свертки
        for k in range(L):
            window += x_extended[k] * y_extended[k] # сумма всех (x*y)
        s_convolved[n] = window 

        # сдвиг y вправо на 1, для смещения окна перекрытия
        for k in range(L-1,0,-1):
            y_extended[k] = y_extended[k-1]
        y_extended[0] = 0j
    return s_convolved


def conv_fft(x: List[complex], y: List[complex]) -> List[complex]:
    """Свёртка с использованием БПФ_В."""
    N = len(x)
    M = len(y)

    # длина БПФ
    L = N + M - 1
    N_fft = 1
    while N_fft < L:
        N_fft *= 2
    
    # дописать нули до конца x
    x_extended = [0j] * N_fft
    for n in range(N):
        x_extended[n] = x[n]
    # дописать нули до конца y
    y_extended = [0j] * N_fft
    for n in range(M):
        y_extended[n] = y[n]

    # FFT для обеих функций
    X_fft = fft_dit(x_extended)
    Y_fft = fft_dit(y_extended)

    # поэлементное умножение спектров
    Z_fft = [0j] * N_fft
    for n in range(N_fft):
        Z_fft[n] = X_fft[n] * Y_fft[n]

    # IFFT для z
    z_ifft = ifft_dit(Z_fft)

    # выбрать первые L отсчетов
    s_convolved = [0j] * L
    for n in range(L):
        s_convolved[n] = z_ifft[n]
    return s_convolved
