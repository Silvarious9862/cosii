import numpy as np
import math
from typing import List
from dsp_basic import conv_fft

def design_iir_hp_onepole(
    cutoff_hz: float,
    sample_rate_hz: float,
) -> tuple[list[float], list[float]]:
    """
    Однополюсный ВЧ-фильтр первого порядка.
    cutoff_hz      — частота среза (0 < fc < fs/2)
    sample_rate_hz — частота дискретизации
    Возвращает (a, b), где:
      a = [a0, a1], b = [1, -b1]
      y[n] = a0 * x[n] + a1 * x[n-1] - b1 * y[n-1]
    """
    normalized_fc = cutoff_hz / sample_rate_hz   
    alpha = math.exp(-2.0 * math.pi * normalized_fc)

    a0 = (1.0 + alpha) / 2.0
    a1 = -(1.0 - alpha) / 2.0
    b1 = alpha

    a = [a0, a1]
    b = [1.0, -b1]
    return a, b

def apply_iir(
    x: List[float],
    a: List[float],
    b: List[float],
) -> List[float]:
    """
    Применение IIR-фильтра первого порядка:
      y[n] = a0 * x[n] + a1 * x[n-1] - b1 * y[n-1]
    Предполагается a = [a0, a1], b = [1, -b1].
    """
    a0, a1 = a
    _, minus_b1 = b
    b1 = -minus_b1

    y: List[float] = [0.0] * len(x)

    for n in range(len(x)):
        x0 = x[n]
        x1 = x[n - 1] if n - 1 >= 0 else 0.0
        y1 = y[n - 1] if n - 1 >= 0 else 0.0

        y[n] = a0 * x0 + a1 * x1 - b1 * y1

    return y