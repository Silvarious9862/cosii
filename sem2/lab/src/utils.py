import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_image(
    image: np.ndarray,
    title: str = "",
    cmap: str | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """Отображает изображение OpenCV BGR или одноканальную маску."""
    plt.figure(figsize=figsize)

    if image.ndim == 2:
        plt.imshow(image, cmap=cmap or "gray")
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)

    plt.title(title)
    plt.axis("off")
    plt.show()