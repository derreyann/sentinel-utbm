"""
Utilities used by example notebooks
"""

from __future__ import annotations

from typing import Any

import os

import rasterio
import matplotlib.pyplot as plt
import numpy as np


def plot_image(
    image: np.ndarray,
    factor: float = 1.0,
    clip_range: tuple[float, float] | None = None,
    **kwargs: Any,
) -> None:
    """Utility function for plotting RGB images."""
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])


def create_bounding_box(coords: list[tuple]):
    # create a bounding box using the coords of the fire
    min_lat = min([coord[0] for coord in coords])
    max_lat = max([coord[0] for coord in coords])
    min_lon = min([coord[1] for coord in coords])
    max_lon = max([coord[1] for coord in coords])

    bbox_coords = (min_lat, min_lon, max_lat, max_lon)

    # add padding to the bounding box
    padding = 1.6

    bbox_coords = (
        min_lat - padding,
        min_lon - padding,
        max_lat + padding,
        max_lon + padding,
    )
    return bbox_coords
