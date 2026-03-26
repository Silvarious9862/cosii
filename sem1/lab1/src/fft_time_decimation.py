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
