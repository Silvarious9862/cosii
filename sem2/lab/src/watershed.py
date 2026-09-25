import cv2
import numpy as np

def compute_distance_map(binary_mask):
    """
    Строит distance transform и нормализованную версию для отображения.
    """
    distance_map = cv2.distanceTransform(
        binary_mask,
        cv2.DIST_L2,
        5
    )

    normalized = cv2.normalize(
        distance_map,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)

    return distance_map, normalized


def watershed_segmentation(
    image_bgr,
    filled_mask,
    distance_threshold=0.42,
    dilation_iterations=3
):
    """
    Разделяет соприкасающиеся объекты методом watershed.

    Важно:
    метод использует заполненную маску, чтобы внутренние символы
    не становились ложными границами.
    """
    distance_map, distance_normalized = compute_distance_map(filled_mask)

    _, sure_foreground = cv2.threshold(
        distance_map,
        distance_threshold * distance_map.max(),
        255,
        cv2.THRESH_BINARY
    )

    sure_foreground = sure_foreground.astype(np.uint8)

    kernel = np.ones((3, 3), dtype=np.uint8)

    sure_background = cv2.dilate(
        filled_mask,
        kernel,
        iterations=dilation_iterations
    )

    unknown = cv2.subtract(sure_background, sure_foreground)

    marker_count, markers = cv2.connectedComponents(sure_foreground)

    markers = markers + 1
    markers[unknown == 255] = 0

    watershed_markers = markers.copy()
    cv2.watershed(image_bgr.copy(), watershed_markers)

    return {
        "distance_map": distance_map,
        "distance_normalized": distance_normalized,
        "sure_foreground": sure_foreground,
        "sure_background": sure_background,
        "unknown": unknown,
        "initial_markers": markers,
        "watershed_markers": watershed_markers,
        "marker_count": marker_count - 1,
    }


def colorize_markers(markers, seed=42):
    """
    Преобразует номера watershed-областей в цветное диагностическое изображение.
    """
    result = np.zeros((*markers.shape, 3), dtype=np.uint8)

    labels = np.unique(markers)

    rng = np.random.default_rng(seed)

    colors = {
        label: rng.integers(60, 256, size=3, dtype=np.uint8)
        for label in labels
        if label > 1
    }

    for label, color in colors.items():
        result[markers == label] = color

    result[markers == -1] = (0, 0, 255)

    return result

def contours_from_watershed(
    watershed_markers,
    min_area=800
):
    """
    Извлекает отдельные внешние контуры из меток watershed.

    Метки:
    -1 — границы watershed;
     1 — фон;
    >1 — отдельные области объектов.

    Возвращает список контуров подходящей площади.
    """
    contours = []

    object_labels = np.unique(watershed_markers)
    object_labels = object_labels[object_labels > 1]

    for label in object_labels:
        object_mask = np.zeros(
            watershed_markers.shape,
            dtype=np.uint8
        )

        object_mask[watershed_markers == label] = 255

        current_contours, _ = cv2.findContours(
            object_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in current_contours:
            if cv2.contourArea(contour) >= min_area:
                contours.append(contour)

    return contours

def extract_watershed_regions(markers, min_area=8_000):
    """
    Извлекает отдельные объекты из разметки watershed.

    Метки:
    -1 — линия водораздела;
     1 — фон;
    >1 — области объектов.
    """
    regions = []

    for label in np.unique(markers):
        if label <= 1:
            continue

        region_mask = np.zeros(markers.shape, dtype=np.uint8)
        region_mask[markers == label] = 255

        contours, _ = cv2.findContours(
            region_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        moments = cv2.moments(contour)

        if moments["m00"] != 0:
            center = (
                moments["m10"] / moments["m00"],
                moments["m01"] / moments["m00"],
            )
        else:
            x, y, width, height = cv2.boundingRect(contour)
            center = (x + width / 2, y + height / 2)

        regions.append({
            "label": int(label),
            "mask": region_mask,
            "contour": contour,
            "area": float(area),
            "perimeter": float(perimeter),
            "center": center,
        })

    return regions