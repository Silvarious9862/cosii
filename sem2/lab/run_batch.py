from __future__ import annotations

import csv
from pathlib import Path

import cv2

from src.classification import classify_regions
from src.config import PipelineConfig
from src.io import (
    load_image,
    resize_for_processing,
    resize_mask_to_shape,
    scale_prediction,
)
from src.segmentation import automatic_kmeans_mask
from src.filtering import filter_chip_regions
from src.visualization import draw_final_result
from src.watershed import (
    extract_watershed_regions,
    watershed_segmentation,
)


INPUT_DIR = Path("images")
OUTPUT_DIR = Path("output")
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
REPORT_PATH = OUTPUT_DIR / "batch_report.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(directory: Path) -> list[Path]:
    """Возвращает отсортированный список входных изображений."""
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def process_image(
    image_path: Path,
    config: PipelineConfig,
) -> tuple[object, dict]:
    """
    Выполняет полный конвейер для одного изображения.

    Возвращает:
    - итоговое BGR-изображение с разметкой;
    - строку для итогового отчёта.
    """
    image_bgr = load_image(image_path)

    working_image_bgr = resize_for_processing(
        image_bgr,
        max_width=config.max_width,
    )

    processing_image_bgr = cv2.GaussianBlur(
        working_image_bgr,
        (5, 5),
        0,
    )

    best_mask_result, _ = automatic_kmeans_mask(
        processing_image_bgr,
        k_values=config.k_values,
        attempts=config.kmeans_attempts,
        seed=config.kmeans_seed,
        border_width=config.border_width,
        min_border_fraction=config.min_border_fraction,
        kernel_size=config.morphology_kernel_size,
        open_iterations=config.open_iterations,
        close_iterations=config.close_iterations,
        min_component_area=config.min_component_area,
    )

    filled_mask = best_mask_result["filled_mask"]

    watershed_result = watershed_segmentation(
        working_image_bgr,
        filled_mask,
        distance_threshold=config.distance_threshold,
        dilation_iterations=config.dilation_iterations,
    )

    regions = extract_watershed_regions(
        watershed_result["watershed_markers"],
        min_area=config.min_region_area,
    )
    
    regions = filter_chip_regions(
        regions,
        max_area_ratio=config.max_chip_area_ratio,
        max_aspect_ratio=config.max_chip_aspect_ratio,
    )

    predictions = classify_regions(
        regions,
        epsilon_ratio=config.approximation_epsilon_ratio,
    )

    predictions_original = [
        scale_prediction(
            prediction,
            source_shape=working_image_bgr.shape,
            target_shape=image_bgr.shape,
        )
        for prediction in predictions
    ]

    chip_mask_original = resize_mask_to_shape(
        filled_mask,
        image_bgr.shape,
    )

    final_result_bgr = draw_final_result(
        image=image_bgr,
        predictions=predictions_original,
        chip_mask=chip_mask_original,
        observed_thickness=10,
        predicted_thickness=15,
        font_scale=1.55,
        text_thickness=4,
    )

    shape_counts = {
        "Circle": 0,
        "Square": 0,
        "Hexagon": 0,
    }

    for prediction in predictions:
        shape_counts[prediction["shape"]] += 1

    report_row = {
        "filename": image_path.name,
        "selected_k": best_mask_result["k"],
        "regions_count": len(regions),
        "chips_count": len(predictions),
        "circles_count": shape_counts["Circle"],
        "squares_count": shape_counts["Square"],
        "hexagons_count": shape_counts["Hexagon"],
    }

    return final_result_bgr, report_row


def save_report(rows: list[dict], report_path: Path) -> None:
    """Сохраняет сводку пакетной обработки в CSV."""
    fieldnames = [
        "filename",
        "selected_k",
        "regions_count",
        "chips_count",
        "circles_count",
        "squares_count",
        "hexagons_count",
    ]

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Запускает пакетную обработку всех изображений в INPUT_DIR."""
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Каталог с исходными изображениями не найден: "
            f"{INPUT_DIR.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(INPUT_DIR)

    if not image_paths:
        raise FileNotFoundError(
            f"В каталоге {INPUT_DIR.resolve()} не найдены изображения."
        )

    config = PipelineConfig()
    report_rows = []

    print(f"Найдено изображений: {len(image_paths)}")

    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] Обработка: {image_path.name}")

        try:
            result_bgr, report_row = process_image(
                image_path,
                config,
            )

            output_path = ANNOTATED_DIR / (
                f"{image_path.stem}_annotated.png"
            )

            saved = cv2.imwrite(
                str(output_path),
                result_bgr,
            )

            if not saved:
                raise OSError(
                    f"Не удалось сохранить изображение: "
                    f"{output_path.resolve()}"
                )

            report_rows.append(report_row)

            print(
                "  Готово: "
                f"фишек={report_row['chips_count']}, "
                f"кругов={report_row['circles_count']}, "
                f"квадратов={report_row['squares_count']}, "
                f"шестигранников={report_row['hexagons_count']}"
            )

        except Exception as error:
            print(f"  Ошибка: {error}")

            report_rows.append(
                {
                    "filename": image_path.name,
                    "selected_k": "",
                    "regions_count": "",
                    "chips_count": "",
                    "circles_count": "",
                    "squares_count": "",
                    "hexagons_count": "",
                }
            )

    save_report(report_rows, REPORT_PATH)

    successful_count = sum(
        bool(row["filename"]) and row["chips_count"] != ""
        for row in report_rows
    )

    print()
    print(
        f"Обработка завершена: "
        f"{successful_count}/{len(image_paths)} изображений."
    )
    print(f"Размеченные изображения: {ANNOTATED_DIR.resolve()}")
    print(f"CSV-отчёт: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()