# src/chip_vision/config.py
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PipelineConfig:
    max_width: int = 1200

    k_values: range = field(default_factory=lambda: range(3, 5))
    kmeans_attempts: int = 5
    kmeans_seed: int = 42
    border_width: int = 20
    min_border_fraction: float = 0.08

    morphology_kernel_size: int = 3
    open_iterations: int = 1
    close_iterations: int = 2
    min_component_area: int = 800

    distance_threshold: float = 0.5
    dilation_iterations: int = 3
    min_region_area: int = 8000
    max_chip_area_ratio: float = 2.2
    max_chip_aspect_ratio: float = 1.35

    approximation_epsilon_ratio: float = 0.018

    observed_contour_thickness: int = 3
    predicted_contour_thickness: int = 6
    label_font_scale: float = 1.35
    label_text_thickness: int = 3


SHAPE_LABELS = {
    "Circle": "Круг",
    "Square": "Квадрат",
    "Hexagon": "Шестигранник",
}

SHAPE_COLORS = {
    "Circle": (0, 150, 230),
    "Square": (0, 0, 255),
    "Hexagon": (255, 35, 255),
}