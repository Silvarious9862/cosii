from dataclasses import asdict

import numpy as np

from .classification import classify_regions
from .config import PipelineConfig
from .io import resize_for_processing
from .segmentation import automatic_kmeans_mask
from .visualization import draw_final_result
from .watershed import extract_watershed_regions, watershed_segmentation


def detect_chips(
    image_bgr: np.ndarray,
    config: PipelineConfig | None = None,
) -> dict:
    """Находит фишки, классифицирует их форму и строит итоговую разметку."""
    config = config or PipelineConfig()

    working_image = resize_for_processing(
        image_bgr,
        max_width=config.max_width,
    )

    mask_result, _ = automatic_kmeans_mask(
        working_image,
        k_values=config.k_values,
        attempts=config.kmeans_attempts,
        border_width=config.border_width,
        kernel_size=config.morphology_kernel_size,
        min_component_area=config.min_component_area,
    )

    watershed_result = watershed_segmentation(
        image_bgr=working_image,
        filled_mask=mask_result["filled_mask"],
        distance_threshold=config.distance_threshold,
        dilation_iterations=config.dilation_iterations,
    )

    regions = extract_watershed_regions(
        markers=watershed_result["watershed_markers"],
        min_area=config.min_region_area,
    )

    predictions = classify_regions(
        regions=regions,
        epsilon_ratio=config.approximation_epsilon_ratio,
    )

    result_image = draw_final_result(
        image_bgr=working_image,
        predictions=predictions,
        chip_mask=mask_result["filled_mask"],
        observed_thickness=config.observed_contour_thickness,
        predicted_thickness=config.predicted_contour_thickness,
        font_scale=config.label_font_scale,
        text_thickness=config.label_text_thickness,
    )

    return {
        "result_image": result_image,
        "predictions": predictions,
        "working_image": working_image,
        "foreground_mask": mask_result["foreground_mask"],
        "filled_mask": mask_result["filled_mask"],
        "watershed_markers": watershed_result["watershed_markers"],
        "selected_k": mask_result["k"],
        "config": asdict(config),
    }