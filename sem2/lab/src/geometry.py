# orientation_from_min_area_rect(...)
# best_hexagon_orientation(...)
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ShapeModel:
    shape: str
    center: tuple[float, float]
    radius: float
    vertices: np.ndarray | None = None
    angle: float | None = None
    fit_error: float | None = None
    
def make_regular_polygon_contour(
    center,
    radius,
    sides,
    angle_degrees=0,
):
    """
    Создаёт контур правильного многоугольника в формате OpenCV.

    sides=4 — квадрат.
    sides=6 — правильный шестигранник.
    """
    cx, cy = center

    angles = np.deg2rad(
        angle_degrees + np.arange(sides) * 360 / sides
    )

    points = np.column_stack([
        cx + radius * np.cos(angles),
        cy + radius * np.sin(angles),
    ])

    return np.round(points).astype(np.int32).reshape(-1, 1, 2)


def contour_center(contour):
    """Центр контура по моментам."""
    moments = cv2.moments(contour)

    if abs(moments["m00"]) > 1e-8:
        return (
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        )

    x, y, width, height = cv2.boundingRect(contour)

    return (
        x + width / 2,
        y + height / 2,
    )


def contour_features_fast(contour, epsilon_ratio=0.018):
    """
    Быстро вычисляет геометрические признаки контура.

    Контур сначала заменяется выпуклой оболочкой:
    так внутренняя вогнутая граница watershed меньше влияет
    на число вершин, площадь и круглотность.
    """
    hull = cv2.convexHull(contour)

    area = cv2.contourArea(hull)
    perimeter = cv2.arcLength(hull, True)

    epsilon = epsilon_ratio * perimeter

    approx = cv2.approxPolyDP(
        hull,
        epsilon,
        True,
    )

    circularity = (
        4 * np.pi * area / (perimeter ** 2)
        if perimeter > 0
        else 0.0
    )

    rect = cv2.minAreaRect(hull)
    width, height = rect[1]

    aspect_ratio = (
        min(width, height) / max(width, height)
        if min(width, height) > 0
        else 0.0
    )

    return {
        "hull": hull,
        "area": float(area),
        "perimeter": float(perimeter),
        "approx": approx,
        "vertices": int(len(approx)),
        "circularity": float(circularity),
        "rect": rect,
        "aspect_ratio": float(aspect_ratio),
    }
    
def orientation_from_min_area_rect(rect):
    """
    Нормализует угол minAreaRect к направлению длинной оси объекта.

    Возвращаемый угол используется только как начальная оценка
    для поиска поворота регулярного многоугольника.
    """
    width, height = rect[1]
    angle = float(rect[2])

    if width < height:
        angle += 90.0

    return angle % 180.0

def best_hexagon_orientation(
    hull,
    center,
    radius,
    base_angle=0,
    step_degrees=2,
):
    """
    Выбирает ориентацию шестигранника, минимизирующую
    расстояние от выпуклой оболочки до его сторон.

    Достаточно диапазона 0..60°, поскольку поворот на 60°
    для правильного шестигранника эквивалентен исходному.
    """
    points = hull.reshape(-1, 2).astype(np.float32)

    best = None

    for delta in range(0, 60, step_degrees):
        angle = base_angle + delta

        vertices = make_regular_polygon_contour(
            center=center,
            radius=radius,
            sides=6,
            angle_degrees=angle,
        ).reshape(-1, 2).astype(np.float32)

        error = mean_distance_to_polygon_vectorized(
            points,
            vertices,
        )

        if best is None or error < best["error"]:
            best = {
                "angle": float(angle),
                "vertices": vertices,
                "error": float(error),
            }

    return best

def mean_distance_to_polygon_vectorized(
    points: np.ndarray,
    vertices: np.ndarray,
) -> float:
    """
    Вычисляет среднее расстояние точек до сторон замкнутого многоугольника.

    points имеет форму N×2.
    vertices имеет форму M×2.
    """
    starts = vertices
    ends = np.roll(vertices, shift=-1, axis=0)

    segments = ends - starts
    segment_lengths_sq = np.sum(segments ** 2, axis=1)

    vectors = points[:, None, :] - starts[None, :, :]

    t = np.sum(
        vectors * segments[None, :, :],
        axis=2,
    )

    t /= np.maximum(segment_lengths_sq[None, :], 1e-8)
    t = np.clip(t, 0.0, 1.0)

    nearest_points = (
        starts[None, :, :]
        + t[:, :, None] * segments[None, :, :]
    )

    distances = np.linalg.norm(
        points[:, None, :] - nearest_points,
        axis=2,
    )

    return float(np.mean(np.min(distances, axis=1)))