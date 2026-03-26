import numpy as np
from typing import List
from dsp_basic import conv_fft

def design_fir_hp_rect(
    cutoff_hz: float,
    sample_rate_hz: float,
    filter_length: int
) -> List[float]:
    """
    КИХ ВЧ‑фильтр (метод окон, прямоугольное окно).
    cutoff_hz      — частота среза (Гц) - fc
    sample_rate_hz — частота дискретизации (Гц) - fs
    filter_length  — длина фильтра (нечётное число) - N
    Возвращает импульсную характеристику h[n] в виде списка.
    """

    # Нормализованная угловая частота среза ωc
    omega_c = 2 * np.pi * cutoff_hz / sample_rate_hz

    # Центр симметрии (для нечётного filter_length)
    center_index = (filter_length - 1) // 2

    # Идеальная импульсная характеристика НЧ-фильтра 
    lowpass_impulse: List[float] = [0.0] * filter_length

    for n in range(filter_length):
        k = n - center_index  # смещение относительно центра

        if k == 0:
            # особая точка: h_lp[M] = ωc / π
            lowpass_impulse[n] = omega_c / np.pi
        else:
            # h_lp[n] = sin(ωc * k) / (π * k)
            lowpass_impulse[n] = np.sin(omega_c * k) / (np.pi * k)

    # Идеальный ВЧ-фильтр: delta[n-center] - lowpass_impulse[n]
    highpass_impulse: List[float] = [0.0] * filter_length
    for n in range(filter_length):
        delta = 1.0 if n == center_index else 0.0
        highpass_impulse[n] = delta - lowpass_impulse[n]

    # Прямоугольное окно: w[n] = 1, поэтому просто возвращаем highpass_impulse
    return highpass_impulse

def apply_fir(x: List[float], h: List[float]) -> List[float]:
    """
    Применение MA-фильтра к сигналу x.
    """
    y_full = conv_fft(x, h)           # List[complex]
    # берём действительную часть
    y_full_real = [z.real for z in y_full]
    # подгоняем под длину входного сигнала
    y = y_full_real[:len(x)]
    return y
