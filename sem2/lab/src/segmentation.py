import cv2
import numpy as np


def segment_kmeans(
    image_bgr,
    k=5,
    attempts=10,
    seed=42,
):
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
    
    cv2.setRNGSeed(seed)

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
    k_values=range(3, 5),
    attempts=5,
    seed=42,
    border_width=20,
    min_border_fraction=0.08,
    kernel_size=3,
    open_iterations=1,
    close_iterations=2,
    min_component_area=800,
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
            attempts=attempts,
            seed=seed + k,
        )

        foreground_mask, background_ids = build_foreground_mask(
            label_map,
            border_width=border_width,
            min_border_fraction=min_border_fraction,
        )

        cleaned_mask = clean_mask(
            foreground_mask,
            kernel_size=kernel_size,
            open_iterations=open_iterations,
            close_iterations=close_iterations,
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

