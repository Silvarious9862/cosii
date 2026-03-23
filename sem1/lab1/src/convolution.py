from typing import List


def conv_time(x: List[complex], y: List[complex]) -> List[complex]:
    """Свёртка по определению во временной области."""
    raise NotImplementedError


def conv_fft(x: List[complex], y: List[complex]) -> List[complex]:
    """Свёртка с использованием БПФ (БПФ_В или numpy.fft)."""
    raise NotImplementedError
