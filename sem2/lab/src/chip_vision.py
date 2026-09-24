import cv2
import numpy as np


def show_image(image, title="", cmap=None, figsize=(12, 8)):
    """
    Показывает изображение в Jupyter Notebook.

    Цветное изображение OpenCV в формате BGR преобразуется в RGB.
    Одноканальная маска выводится в оттенках серого.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)

    if image.ndim == 2:
        plt.imshow(image, cmap=cmap or "gray")
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)

    plt.title(title)
    plt.axis("off")
    plt.show()


def resize_for_processing(image_bgr, max_width=1200):
    """
    Уменьшает изображение до заданной ширины, сохраняя пропорции.

    Если исходное изображение уже уже max_width, возвращает его копию.
    """
    height, width = image_bgr.shape[:2]

    if width <= max_width:
        return image_bgr.copy()

    scale = max_width / width
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        image_bgr,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


def segment_kmeans(image_bgr, k=5, attempts=10):
    """
    Сегментирует BGR-изображение методом k-средних.

    Возвращает:
        segmented_bgr — изображение с заменой пикселей центрами кластеров;
        label_map — карта меток кластеров H × W;
        centers_bgr — средние BGR-цвета кластеров;
        compactness — сумма квадратов расстояний пикселей до центров.
    """
    height, width = image_bgr.shape[:2]

    data = np.float32(image_bgr.reshape((-1, 3)))

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1.0
    )

    compactness, labels, centers = cv2.kmeans(
        data,
        K=k,
        bestLabels=None,
        criteria=criteria,
        attempts=attempts,
        flags=cv2.KMEANS_PP_CENTERS
    )

    centers_bgr = np.uint8(centers)

    segmented_bgr = centers_bgr[labels.flatten()]
    segmented_bgr = segmented_bgr.reshape((height, width, 3))

    label_map = labels.reshape((height, width))

    return segmented_bgr, label_map, centers_bgr, compactness


def detect_background_clusters(
    label_map,
    border_width=20,
    min_border_fraction=0.08,
):
    """
    Возвращает все кластеры, заметно представленные на рамке кадра.

    Для текстурного фона дерево может быть разбито k-means
    на несколько кластеров: светлое дерево, тёмное дерево, тени.
    Такие кластеры считаются фоном одновременно.

    min_border_fraction — минимальная доля пикселей рамки,
    которую должен занимать кластер, чтобы считаться фоновым.
    """
    height, width = label_map.shape

    border_width = max(
        1,
        min(border_width, height // 4, width // 4)
    )

    border_labels = np.concatenate([
        label_map[:border_width, :].ravel(),
        label_map[-border_width:, :].ravel(),
        label_map[:, :border_width].ravel(),
        label_map[:, -border_width:].ravel(),
    ])

    cluster_ids, counts = np.unique(
        border_labels,
        return_counts=True,
    )

    border_fractions = counts / counts.sum()

    background_ids = cluster_ids[
        border_fractions >= min_border_fraction
    ]

    return background_ids.astype(int).tolist()

def build_foreground_mask(
    label_map,
    border_width=20,
    min_border_fraction=0.08,
):
    """
    Строит маску переднего плана с несколькими фоновыми кластерами.

    Белое — кластеры, которые не относятся к фону.
    Чёрное — все кластеры, заметно представленные на границе кадра.

    Возвращает:
    - foreground_mask;
    - background_ids: список фоновых кластеров.
    """
    background_ids = detect_background_clusters(
        label_map,
        border_width=border_width,
        min_border_fraction=min_border_fraction,
    )

    foreground_mask = np.uint8(
        ~np.isin(label_map, background_ids)
    ) * 255

    return foreground_mask, background_ids


def fill_holes(binary_mask):
    """
    Заполняет чёрные замкнутые отверстия внутри белых объектов.

    Это устраняет из маски внутренние рисунки/цифры на фишках,
    если они были отнесены k-средних к другому кластеру.
    """
    height, width = binary_mask.shape

    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    flood_filled = binary_mask.copy()

    cv2.floodFill(
        flood_filled,
        flood_mask,
        seedPoint=(0, 0),
        newVal=255
    )

    holes = cv2.bitwise_not(flood_filled)

    return cv2.bitwise_or(binary_mask, holes)


def clean_mask(binary_mask, kernel_size=3, open_iterations=1, close_iterations=2):
    """
    Удаляет мелкие шумы и закрывает небольшие разрывы в бинарной маске.
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    mask_opened = cv2.morphologyEx(
        binary_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=open_iterations
    )

    return cv2.morphologyEx(
        mask_opened,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=close_iterations
    )


def mask_quality_score(binary_mask, min_component_area=800):
    """
    Оценивает пригодность маски для поиска объектов.

    Меньший score означает более предпочтительную маску.
    Возвращает score и диагностические показатели.
    """
    height, width = binary_mask.shape
    foreground_ratio = np.count_nonzero(binary_mask) / (height * width)

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8
    )

    component_areas = stats[1:, cv2.CC_STAT_AREA]

    large_components = component_areas[
        component_areas >= min_component_area
    ]

    large_component_count = len(large_components)

    coverage_penalty = 0.0

    if foreground_ratio < 0.01:
        coverage_penalty += 10.0

    if foreground_ratio > 0.65:
        coverage_penalty += 10.0

    component_penalty = 0.15 * large_component_count
    foreground_penalty = 2.0 * foreground_ratio

    score = coverage_penalty + component_penalty + foreground_penalty

    diagnostics = {
        "foreground_ratio": foreground_ratio,
        "large_components": large_component_count,
    }

    return score, diagnostics


def automatic_kmeans_mask(
    image_bgr,
    k_values=range(3, 9),
    attempts=5,
    border_width=20,
    kernel_size=3,
    min_component_area=800
):
    """
    Автоматически выбирает K из указанного диапазона.

    Для каждого K:
    1. Выполняется k-средних.
    2. Фон определяется по рамке изображения.
    3. Формируется маска всех не-фоновых кластеров.
    4. Маска очищается и заполняются внутренние отверстия.
    5. Рассчитывается эвристическая оценка качества.

    Возвращает словарь лучшего результата и результаты для всех K.
    """
    results = []

    for k in k_values:
        segmented, label_map, centers, compactness = segment_kmeans(
            image_bgr,
            k=k,
            attempts=attempts
        )

        foreground_mask, background_ids = build_foreground_mask(
            label_map,
            border_width=border_width,
            min_border_fraction=0.08,
        )

        cleaned_mask = clean_mask(
            foreground_mask,
            kernel_size=kernel_size
        )

        filled_mask = fill_holes(cleaned_mask)

        score, diagnostics = mask_quality_score(
            filled_mask,
            min_component_area=min_component_area
        )

        results.append({
            "k": k,
            "score": score,
            "segmented": segmented,
            "labels": label_map,
            "centers": centers,
            "compactness": compactness,
            "background_ids": background_ids,
            "foreground_mask": foreground_mask,
            "cleaned_mask": cleaned_mask,
            "filled_mask": filled_mask,
            "diagnostics": diagnostics,
        })

    best_result = min(results, key=lambda result: result["score"])

    return best_result, results


def find_candidate_contours(binary_mask, min_area=800):
    """
    Находит внешние контуры белых объектов и отбрасывает слишком маленькие.
    """
    contours, hierarchy = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= min_area
    ]

    return candidates


def contour_features(contour, epsilon_ratio=0.02):
    """
    Вычисляет геометрические признаки внешнего контура.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    epsilon = epsilon_ratio * perimeter
    polygon = cv2.approxPolyDP(contour, epsilon, True)

    x, y, width, height = cv2.boundingRect(contour)

    circularity = 0.0
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)

    aspect_ratio = width / height if height > 0 else 0.0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    solidity = area / hull_area if hull_area > 0 else 0.0

    return {
        "area": area,
        "perimeter": perimeter,
        "polygon": polygon,
        "vertices": len(polygon),
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "bbox": (x, y, width, height),
    }


def classify_shape(
    features,
    min_area=10_000,
    min_solidity=0.90,
    min_circle_circularity=0.85
):
    """
    Определяет класс формы только по геометрическим признакам.

    Возможные результаты:
        Круг, Квадрат, Шестигранник, Не фишка.
    """
    area = features["area"]
    solidity = features["solidity"]
    circularity = features["circularity"]
    vertices = features["vertices"]

    if area < min_area:
        return "Не фишка"

    if solidity < min_solidity:
        return "Не фишка"

    if circularity >= min_circle_circularity:
        return "Круг"

    if vertices == 4:
        return "Квадрат"

    if vertices == 6:
        return "Шестигранник"

    return "Не фишка"


def classify_contours(
    contours,
    epsilon_ratio=0.02,
    min_area=10_000,
    min_solidity=0.90,
    min_circle_circularity=0.85
):
    """
    Рассчитывает признаки и классифицирует все переданные контуры.
    """
    predictions = []

    for object_id, contour in enumerate(contours, start=1):
        features = contour_features(
            contour,
            epsilon_ratio=epsilon_ratio
        )

        shape = classify_shape(
            features,
            min_area=min_area,
            min_solidity=min_solidity,
            min_circle_circularity=min_circle_circularity
        )

        predictions.append({
            "object_id": object_id,
            "contour": contour,
            "features": features,
            "shape": shape,
        })

    return predictions


def draw_classification(
    image_bgr,
    predictions,
    contour_thickness=7,
    text_scale=1.15,
    text_thickness=3
):
    """
    Рисует внешние контуры и английские подписи только для распознанных фишек.

    Объекты класса 'Не фишка' намеренно не отображаются.
    """
    result = image_bgr.copy()

    shape_colors = {
        "Круг": (0, 255, 0),
        "Квадрат": (0, 255, 255),
        "Шестигранник": (255, 0, 255),
    }

    shape_names = {
        "Круг": "Circle",
        "Квадрат": "Square",
        "Шестигранник": "Hexagon",
    }

    for item in predictions:
        shape = item["shape"]

        if shape == "Не фишка":
            continue

        contour = item["contour"]
        x, y, width, height = item["features"]["bbox"]

        color = shape_colors[shape]

        cv2.drawContours(
            result,
            [contour],
            contourIdx=-1,
            color=color,
            thickness=contour_thickness
        )

        cv2.putText(
            result,
            shape_names[shape],
            (x, max(y - 18, 35)),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            color,
            text_thickness,
            cv2.LINE_AA
        )

    return result


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


def scale_contours(contours, source_shape, target_shape):
    """
    Масштабирует контуры из координат source_shape в координаты target_shape.

    source_shape и target_shape передаются в формате image.shape[:2]:
    (высота, ширина).
    """
    source_height, source_width = source_shape[:2]
    target_height, target_width = target_shape[:2]

    scale_x = target_width / source_width
    scale_y = target_height / source_height

    scaled_contours = []

    for contour in contours:
        scaled = contour.astype(np.float32).copy()

        scaled[:, 0, 0] *= scale_x
        scaled[:, 0, 1] *= scale_y

        scaled_contours.append(
            np.round(scaled).astype(np.int32)
        )

    return scaled_contours