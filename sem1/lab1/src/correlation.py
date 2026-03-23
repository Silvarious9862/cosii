from typing import List


def corr_time(x: List[complex], y: List[complex]) -> List[complex]:
    """Корреляция по определению во временной области."""
    raise NotImplementedError


def corr_fft(x: List[complex], y: List[complex]) -> List[complex]:
    """Корреляция через БПФ по теореме корреляции."""
    raise NotImplementedError
