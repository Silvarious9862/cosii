import numpy as np
from typing import Tuple


def generate_signals(fs: int = 44100, duration: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Генерация двух тестовых периодических сигналов и временной оси.

    Возвращает t, x, y.
    """
    t = np.arange(0, duration, 1.0 / fs)
    x = 2 * np.sin(3 * t) + np.cos(2 * t + 1)
    y = np.sin(5 * t) + 0.5 * np.cos(7 * t)
    return t, x.astype(np.float64), y.astype(np.float64)
