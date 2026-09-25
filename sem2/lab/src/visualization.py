# draw_dashed_segment(...)
# draw_dashed_circle(...)
# draw_dashed_polygon(...)
# draw_predicted_model(...)
# draw_observed_watershed_contour(...)

# label_box(...)
# clip_label_point(...)
# boxes_intersect(...)
# line_hits_mask(...)
# choose_label_position(...)
# draw_label_with_leader(...)

# draw_final_result(...)
import cv2
import numpy as np

from .config import SHAPE_COLORS, SHAPE_LABELS

def draw_dashed_segment(
    image,
    start,
    end,
    color,
    thickness=2,
    dash_length=16,
    gap_length=24,
):
    """
    Рисует пунктирный отрезок между двумя точками.
    """
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)

    vector = end - start
    length = float(np.linalg.norm(vector))

    if length < 1:
        return

    direction = vector / length
    position = 0.0

    while position < length:
        dash_start = start + direction * position
        dash_end = start + direction * min(
            position + dash_length,
            length,
        )

        cv2.line(
            image,
            tuple(np.round(dash_start).astype(int)),
            tuple(np.round(dash_end).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )

        position += dash_length + gap_length



def draw_dashed_circle(
    image,
    center,
    radius,
    color,
    thickness=4,
    dash_degrees=14,
    gap_degrees=9,
):
    """
    Рисует окружность короткими дугами с промежутками.
    """
    center = tuple(np.round(center).astype(int))
    radius = max(1, int(round(radius)))

    angle = 0

    while angle < 360:
        start_angle = angle
        end_angle = min(angle + dash_degrees, 360)

        cv2.ellipse(
            image,
            center,
            (radius, radius),
            0,
            start_angle,
            end_angle,
            color,
            thickness,
            cv2.LINE_AA,
        )

        angle += dash_degrees + gap_degrees


def draw_dashed_polygon(
    image,
    vertices,
    color,
    thickness=4,
    dash_length=30,
    gap_length=30,
):
    """
    Рисует замкнутый многоугольник пунктирной линией.
    """
    vertices = np.round(vertices).astype(np.int32)

    for index in range(len(vertices)):
        start = vertices[index]
        end = vertices[(index + 1) % len(vertices)]

        draw_dashed_segment(
            image,
            start,
            end,
            color,
            thickness=thickness,
            dash_length=dash_length,
            gap_length=gap_length,
        )
        
def draw_predicted_model(
    image,
    prediction,
    thickness=6,
):
    """
    Рисует предсказанную идеализированную модель фигуры пунктиром.

    Цвет модели соответствует классу:
    зелёный — круг,
    пурпурный — квадрат,
    жёлтый — шестигранник.
    """
    canvas = image.copy()

    model = prediction["model"]
    shape = model["shape"]
    color = SHAPE_COLORS[shape]

    if shape == "Circle":
        draw_dashed_circle(
            canvas,
            center=model["center"],
            radius=model["radius"],
            color=color,
            thickness=thickness,
        )

    else:
        draw_dashed_polygon(
            canvas,
            vertices=model["vertices"],
            color=color,
            thickness=thickness,
        )

    return canvas

def draw_observed_watershed_contour(
    image,
    prediction,
    color=(255, 255, 255),
    thickness=3,
):
    """
    Рисует фактически наблюдаемую границу области watershed.

    Белая линия показывает результат сегментации:
    это не восстановленная геометрическая фигура,
    а реально выделенная область объекта.
    """
    canvas = image.copy()

    cv2.drawContours(
        canvas,
        [prediction["contour"]],
        contourIdx=-1,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    return canvas

def label_box(text, label_point, font_scale, thickness, padding=8):
    """
    Возвращает границы текстовой плашки:
    x1, y1, x2, y2 и координату начала текста.
    """
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )

    x, y = label_point

    x1 = x
    y1 = y - text_height - padding
    x2 = x + text_width + 2 * padding
    y2 = y + baseline + padding

    text_origin = (
        x1 + padding,
        y2 - baseline - padding,
    )

    return x1, y1, x2, y2, text_origin


def clip_label_point(label_point, text, image_shape, font_scale, thickness):
    """Сдвигает плашку так, чтобы она не вышла за границы изображения."""
    height, width = image_shape[:2]

    x1, y1, x2, y2, _ = label_box(
        text,
        label_point,
        font_scale,
        thickness,
    )

    shift_x = 0
    shift_y = 0

    if x1 < 5:
        shift_x = 5 - x1
    elif x2 > width - 5:
        shift_x = width - 5 - x2

    if y1 < 5:
        shift_y = 5 - y1
    elif y2 > height - 5:
        shift_y = height - 5 - y2

    return (
        label_point[0] + shift_x,
        label_point[1] + shift_y,
    )


def boxes_intersect(box_a, box_b):
    """Проверяет пересечение двух прямоугольников."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    return not (
        ax2 < bx1
        or bx2 < ax1
        or ay2 < by1
        or by2 < ay1
    )


def line_hits_mask(start, end, mask, samples=50):
    """
    Проверяет, пересекает ли выноска маску фишек.

    Используется для выбора выноски, проходящей преимущественно
    по фону, а не через фишки.
    """
    height, width = mask.shape

    xs = np.linspace(start[0], end[0], samples).astype(int)
    ys = np.linspace(start[1], end[1], samples).astype(int)

    hits = 0

    for x, y in zip(xs, ys):
        if 0 <= x < width and 0 <= y < height:
            hits += int(mask[y, x] > 0)

    return hits


def choose_label_position(
    text,
    anchor,
    chip_mask,
    occupied_boxes,
    image_shape,
    font_scale=1.55,
    thickness=4,
):
    """
    Выбирает наиболее свободное место для текстовой плашки.

    Приоритет:
    1. Плашка не пересекает ранее размещённые подписи.
    2. Выноска пересекает минимум пикселей маски фишек.
    3. Плашка располагается дальше от центра объекта.
    """
    candidate_offsets = [
        (180, -145),
        (180, 145),
        (-420, -145),
        (-420, 145),
        (20, -240),
        (20, 240),
        (310, -35),
        (-500, -35),
    ]

    best = None

    for offset in candidate_offsets:
        candidate = (
            int(anchor[0] + offset[0]),
            int(anchor[1] + offset[1]),
        )

        candidate = clip_label_point(
            candidate,
            text,
            image_shape,
            font_scale,
            thickness,
        )

        x1, y1, x2, y2, _ = label_box(
            text,
            candidate,
            font_scale,
            thickness,
        )

        box = (x1, y1, x2, y2)

        overlaps = sum(
            boxes_intersect(box, old_box)
            for old_box in occupied_boxes
        )

        line_penalty = line_hits_mask(
            anchor,
            candidate,
            chip_mask,
        )

        distance = float(
            np.hypot(
                candidate[0] - anchor[0],
                candidate[1] - anchor[1],
            )
        )

        score = (
            overlaps * 10_000
            + line_penalty * 100
            - distance * 0.01
        )

        if best is None or score < best["score"]:
            best = {
                "point": candidate,
                "box": box,
                "score": score,
            }

    return best["point"], best["box"]

def draw_label_with_leader(
    image,
    text,
    anchor,
    color,
    offset=(85, -65),
    font_scale=0.85,
    thickness=4,
):
    """
    Рисует выноску: линия от объекта к подписи и прямоугольник текста.
    """
    x, y = map(int, np.round(anchor))
    dx, dy = offset

    label_point = (x + dx, y + dy)

    cv2.line(
        image,
        (x, y),
        label_point,
        color,
        thickness,
        cv2.LINE_AA,
    )

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )

    padding = 8
    x1 = label_point[0]
    y1 = label_point[1] - text_height - padding
    x2 = label_point[0] + text_width + 2 * padding
    y2 = label_point[1] + baseline + padding

    height, width = image.shape[:2]

    shift_x = 0
    shift_y = 0

    if x1 < 0:
        shift_x = -x1
    elif x2 >= width:
        shift_x = width - x2 - 1

    if y1 < 0:
        shift_y = -y1
    elif y2 >= height:
        shift_y = height - y2 - 1

    x1 += shift_x
    x2 += shift_x
    y1 += shift_y
    y2 += shift_y

    text_origin = (
        x1 + padding,
        y2 - baseline - padding,
    )

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        -1,
        cv2.LINE_AA,
    )

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        text,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )



def draw_final_result(
    image,
    predictions,
    chip_mask,
    observed_thickness=3,
    predicted_thickness=6,
    font_scale=1.35,
    text_thickness=3,
):
    """
    Наносит на изображение контуры объектов, геометрические модели
    и подписи с выносками.
    """
    result = image.copy()

    counters = {
        "Circle": 0,
        "Square": 0,
        "Hexagon": 0,
    }

    occupied_boxes = []

    for prediction in predictions:
        shape = prediction["shape"]
        model = prediction["model"]

        color = SHAPE_COLORS[shape]

        counters[shape] += 1

        # Подпись для пользователя: русское название формы.
        text = SHAPE_LABELS[shape]

        # Если хочешь нумеровать одинаковые фигуры, используй вместо строки выше:
        # text = f"{SHAPE_LABELS[shape]} №{counters[shape]}"

        # 1. Идеализированная модель фигуры.
        result = draw_predicted_model(
            result,
            prediction,
            thickness=predicted_thickness,
        )

        # 2. Контур реально выделенной watershed-области.
        result = draw_observed_watershed_contour(
            result,
            prediction,
            color=(235, 235, 235),
            thickness=observed_thickness,
        )

        # 3. Поиск свободного положения для подписи.
        label_point, box = choose_label_position(
            text=text,
            anchor=model["center"],
            chip_mask=chip_mask,
            occupied_boxes=occupied_boxes,
            image_shape=result.shape,
            font_scale=font_scale,
            thickness=text_thickness,
        )

        draw_label_with_leader(
            result,
            text=text,
            anchor=model["center"],
            color=color,
            offset=(
                label_point[0] - int(model["center"][0]),
                label_point[1] - int(model["center"][1]),
            ),
            font_scale=font_scale,
            thickness=text_thickness,
        )

        occupied_boxes.append(box)

    return result