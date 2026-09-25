from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Загружает BGR-изображение и проверяет успешность чтения."""
    path = Path(path)
    image_bgr = cv2.imread(str(path))

    if image_bgr is None:
        raise FileNotFoundError(f"Не удалось открыть изображение: {path}")

    return image_bgr


def resize_for_processing(
    image_bgr: np.ndarray,
    max_width: int = 1200,
) -> np.ndarray:
    """Уменьшает изображение, сохраняя пропорции."""
    height, width = image_bgr.shape[:2]

    if width <= max_width:
        return image_bgr.copy()

    scale = max_width / width
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        image_bgr,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )
    
def scale_contour(contour, scale_x, scale_y):
    """
    Масштабирует OpenCV-контур формы (N, 1, 2)
    из координат рабочего изображения в координаты исходного.
    """
    scaled = contour.astype(np.float32).copy()

    scaled[:, :, 0] *= scale_x
    scaled[:, :, 1] *= scale_y

    return np.round(scaled).astype(np.int32)


def scale_prediction(prediction, source_shape, target_shape):
    """
    Масштабирует данные одного предсказания к другому разрешению.

    Масштабируются:
    - наблюдаемый contour watershed;
    - геометрическая модель: центр, радиус, вершины;
    - диагностические контуры hull и approx, если они понадобятся далее.
    """
    source_height, source_width = source_shape[:2]
    target_height, target_width = target_shape[:2]

    scale_x = target_width / source_width
    scale_y = target_height / source_height

    # Для радиуса используем средний линейный масштаб.
    # В нормальном случае aspect ratio изображения сохраняется,
    # поэтому scale_x и scale_y практически одинаковы.
    scale_r = (scale_x + scale_y) / 2

    scaled = {
        **prediction,
        "model": {
            **prediction["model"],
        },
        "features": {
            **prediction["features"],
        },
    }

    # 1. Реальный контур watershed — это и есть источник белой линии.
    scaled["contour"] = scale_contour(
        prediction["contour"],
        scale_x,
        scale_y,
    )

    # 2. Центр области — полезен для диагностики и подписей.
    center_x, center_y = prediction["center"]

    scaled["center"] = (
        center_x * scale_x,
        center_y * scale_y,
    )

    # 3. Геометрическая модель.
    model_center_x, model_center_y = prediction["model"]["center"]

    scaled["model"]["center"] = (
        model_center_x * scale_x,
        model_center_y * scale_y,
    )

    scaled["model"]["radius"] = (
        prediction["model"]["radius"] * scale_r
    )

    model_vertices = prediction["model"]["vertices"]

    if model_vertices is not None:
        scaled["model"]["vertices"] = np.column_stack([
            model_vertices[:, 0] * scale_x,
            model_vertices[:, 1] * scale_y,
        ]).astype(np.float32)

    # 4. Эти данные напрямую не нужны в финале,
    # но пусть всё предсказание остаётся консистентным.
    if "hull" in prediction["features"]:
        scaled["features"]["hull"] = scale_contour(
            prediction["features"]["hull"],
            scale_x,
            scale_y,
        )

    if "approx" in prediction["features"]:
        scaled["features"]["approx"] = scale_contour(
            prediction["features"]["approx"],
            scale_x,
            scale_y,
        )

    return scaled

def resize_mask_to_shape(mask: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Масштабирует бинарную маску к размеру целевого изображения."""
    target_height, target_width = target_shape[:2]

    return cv2.resize(
        mask,
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )