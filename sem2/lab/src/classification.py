from __future__ import annotations

import cv2
import numpy as np

from .geometry import (
    best_hexagon_orientation,
    contour_center,
    contour_features_fast,
    make_regular_polygon_contour,
    orientation_from_min_area_rect,
)


APPROX_EPSILON_RATIO = 0.018

TEMPLATE_CENTER = (200, 200)
TEMPLATE_RADIUS = 100


def create_shape_templates() -> dict[str, np.ndarray]:
    """
    Создаёт эталонные контуры для сравнения форм через cv2.matchShapes.

    Ключи словаря — внутренние технические идентификаторы классов:
    Circle, Square, Hexagon.
    """
    circle_template = cv2.ellipse2Poly(
        center=TEMPLATE_CENTER,
        axes=(TEMPLATE_RADIUS, TEMPLATE_RADIUS),
        angle=0,
        arcStart=0,
        arcEnd=360,
        delta=5,
    ).reshape(-1, 1, 2)

    square_template = make_regular_polygon_contour(
        center=TEMPLATE_CENTER,
        radius=TEMPLATE_RADIUS,
        sides=4,
        angle_degrees=45,
    )

    hexagon_template = make_regular_polygon_contour(
        center=TEMPLATE_CENTER,
        radius=TEMPLATE_RADIUS,
        sides=6,
        angle_degrees=0,
    )

    return {
        "Circle": circle_template,
        "Square": square_template,
        "Hexagon": hexagon_template,
    }


SHAPE_TEMPLATES = create_shape_templates()


def classify_region(
    region: dict,
    epsilon_ratio: float = APPROX_EPSILON_RATIO,
) -> dict:
    """
    Классифицирует одну watershed-область.

    Решение принимается по совокупности признаков:
    - количеству вершин приближённого контура;
    - круглотности;
    - отношению сторон минимального ограничивающего прямоугольника;
    - сходству с шаблонами cv2.matchShapes.

    Возвращает словарь региона, дополненный полями:
    shape, features, shape_scores, confidence.
    """
    contour = region["contour"]

    features = contour_features_fast(
        contour,
        epsilon_ratio=epsilon_ratio,
    )

    hull = features["hull"]

    shape_scores = {
        shape: cv2.matchShapes(
            hull,
            template,
            cv2.CONTOURS_MATCH_I1,
            0.0,
        )
        for shape, template in SHAPE_TEMPLATES.items()
    }

    best_by_hu = min(shape_scores, key=shape_scores.get)

    vertices = features["vertices"]
    circularity = features["circularity"]
    aspect_ratio = features["aspect_ratio"]

    if circularity >= 0.90 and vertices >= 7:
        shape = "Circle"

    elif vertices == 4 and aspect_ratio >= 0.82:
        shape = "Square"

    elif vertices == 6:
        shape = "Hexagon"

    else:
        shape = best_by_hu

    ordered_scores = sorted(shape_scores.values())
    best_score = ordered_scores[0]
    second_score = ordered_scores[1]

    confidence = 1.0 - best_score / max(second_score, 1e-8)
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return {
        **region,
        "shape": shape,
        "features": features,
        "shape_scores": shape_scores,
        "confidence": confidence,
    }


def model_from_prediction(prediction: dict) -> dict:
    """
    Восстанавливает идеализированную геометрическую модель фигуры.

    Для круга возвращается центр и радиус.
    Для квадрата и шестигранника возвращаются вершины правильного
    многоугольника, ориентированного по найденному объекту.
    """
    shape = prediction["shape"]
    hull = prediction["features"]["hull"]

    center = contour_center(hull)
    _, radius = cv2.minEnclosingCircle(hull)

    if shape == "Circle":
        return {
            "shape": shape,
            "center": (float(center[0]), float(center[1])),
            "radius": float(radius),
            "vertices": None,
        }

    object_angle = orientation_from_min_area_rect(
        prediction["features"]["rect"]
    )

    if shape == "Square":
        vertices = make_regular_polygon_contour(
            center=center,
            radius=radius,
            sides=4,
            angle_degrees=object_angle + 45.0,
        ).reshape(-1, 2).astype(np.float32)

        return {
            "shape": shape,
            "center": (float(center[0]), float(center[1])),
            "radius": float(radius),
            "vertices": vertices,
            "angle": float((object_angle + 45.0) % 90.0),
        }

    hexagon_fit = best_hexagon_orientation(
        hull=hull,
        center=center,
        radius=radius,
        base_angle=object_angle,
        step_degrees=2,
    )

    return {
        "shape": shape,
        "center": (float(center[0]), float(center[1])),
        "radius": float(radius),
        "vertices": hexagon_fit["vertices"].astype(np.float32),
        "angle": float(hexagon_fit["angle"] % 60.0),
        "fit_error": float(hexagon_fit["error"]),
    }


def classify_regions(
    regions: list[dict],
    epsilon_ratio: float = APPROX_EPSILON_RATIO,
) -> list[dict]:
    """
    Классифицирует список watershed-областей и добавляет модель формы.

    Сортировка по вертикали, затем по горизонтали обеспечивает
    воспроизводимый порядок вывода и нумерации подписей.
    """
    predictions = [
        classify_region(
            region,
            epsilon_ratio=epsilon_ratio,
        )
        for region in regions
    ]

    for prediction in predictions:
        prediction["model"] = model_from_prediction(prediction)

    predictions.sort(
        key=lambda prediction: (
            prediction["center"][1],
            prediction["center"][0],
        )
    )

    return predictions