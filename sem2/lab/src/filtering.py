from __future__ import annotations

import cv2
import numpy as np


def filter_chip_regions(
    regions: list[dict],
    *,
    max_area_ratio: float = 2.2,
    max_aspect_ratio: float = 1.35,
) -> list[dict]:
    """
    Исключает регионы, геометрически не похожие на фишки.

    Правила:
    1. Фишки на одном фото имеют сопоставимую площадь.
       Слишком крупные регионы относительно медианной площади исключаются.
    2. Фишки компактны: их minAreaRect не должен быть слишком вытянут.
    """
    if not regions:
        return []

    areas = np.array(
        [region["area"] for region in regions],
        dtype=np.float64,
    )

    median_area = float(np.median(areas))

    filtered_regions = []

    for region in regions:
        if region["area"] > median_area * max_area_ratio:
            continue

        contour = region["contour"]
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]

        if min(width, height) <= 1e-8:
            continue

        aspect_ratio = max(width, height) / min(width, height)

        if aspect_ratio > max_aspect_ratio:
            continue

        filtered_regions.append(region)

    return filtered_regions