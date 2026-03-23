import cmath
from typing import List


def fft_dit(x: List[complex], inverse: bool = False) -> List[complex]:
    """БПФ с прореживанием по времени (Decimation In Time).
    x: список комплексных отсчётов, длина N=2^k.
    inverse=True вычисляет обратное БПФ.
    """
    raise NotImplementedError
