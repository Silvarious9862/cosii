import cmath
from typing import List
import math


def fft_dit(x: List[complex], inverse: bool = False) -> List[complex]:
    """БПФ с прореживанием по времени (Decimation In Time).
    x: список комплексных отсчётов, длина N=2^k.
    inverse=True вычисляет обратное БПФ.
    """
    N = len(x)
    if N==1:
        return x
    # разбить вектор на чет и нечет
    x_even = x[0::2]
    x_odd = x[1::2]
    # рекурсивный вызов 
    B_even = fft_dit(x_even, inverse=inverse)
    B_odd = fft_dit(x_odd, inverse=inverse)

    # направление отсчета
    dir_sign = 1 if inverse else -1
    # коэффициенты W
    omega_N = cmath.exp(dir_sign * 2j * math.pi /N)
    omega = 1+0j

    # результирующий вектор
    y = [0j] * N

    # бабочка
    for j in range(N//2):
        y[j] = B_even[j] + omega * B_odd[j]
        y[j+N//2] = B_even[j] - omega * B_odd[j]
        omega *= omega_N
    return y

def ifft_dit(X: List[complex]) -> List[complex]:
    N = len(X)
    x_rec = fft_dit(X, inverse=True)
    return [val / N for val in x_rec]

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
