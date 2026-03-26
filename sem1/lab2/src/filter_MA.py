from typing import List
from dsp_basic import conv_fft

def design_ma(N: int) -> List[float]:
    """
    Однородный фильтр (скользящее среднее) длины N.
    Возвращает массив коэффициентов h[n].
    """
    h = [1.0] * N
    for value in range(N):
        h[value] = h[value] / N
    return h

def apply_ma(x: List[float], h: List[float]) -> List[float]:
    """
    Применение MA-фильтра к сигналу x.
    """
    y_full = conv_fft(x, h)           # List[complex]
    # берём действительную часть
    y_full_real = [z.real for z in y_full]
    # подгоняем под длину входного сигнала
    y = y_full_real[:len(x)]
    return y
