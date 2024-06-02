from datetime import datetime
import os

import numpy as np
import rasterio
from rasterio.warp import Resampling
import rioxarray as rxr
import xarray as xr


def extract_fire_mask(input_path: str, output_dir: str = "../data/modis/processing") -> tuple[xr.Dataset, list[datetime], str]:
    """
    Extracts the fire mask from a MODIS HDF file, reprojects it to EPSG:4326, 
    and saves it as a GeoTIFF file. Also extracts the dates from the file attributes.

    Parameters:
    input_path (str): The path to the input HDF file.
    output_dir (str, optional): The directory to save the output file. If None, saves in input_dir.

    Returns:
    tuple[xr.Dataset, list[datetime], str]: The reprojected fire mask dataset, list of dates and the full path to the processed file.
    """
    # Retrieve data
    dataset = rxr.open_rasterio(input_path, masked=True)

    # Reproject
    dataset = dataset.rio.reproject("EPSG:4326")

    # Extracts dates
    dates = dataset.attrs["DAYSOFYEAR"]
    dates = dates.split(", ")
    date_objects = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in dates]
    
    dataset.FireMask.rio.write_nodata(0, inplace=True)
    dataset = dataset.FireMask.rio.reproject("EPSG:4326")

    # Save the new file to tiff
    output_filename = os.path.basename(input_path)[:-3] + "fire.tif"
    output_full_path = os.path.join(output_dir, output_filename)
    dataset.rio.to_raster(output_full_path)

    return dataset, date_objects, output_full_path


def get_coords_and_pixels(dataset: rasterio.io.DatasetReader) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    """
    Extracts the coordinates and pixels with high fire confidence from a fire band.

    Parameters:
    dataset (rasterio.io.DatasetReadert): The input rasterio dataset.

    Returns:
    Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]: A tuple containing two lists:
        - List of coordinates (longitude, latitude) of high fire confidence pixels.
        - List of pixel indices (row, column) of high fire confidence pixels.
    """
    # Read the first band of the dataset
    fire_band = dataset.read(1)
    
    # Get the indices of pixels with high fire confidence
    high_confidence_indices = np.argwhere(fire_band > 6)
    
    # Extract the coordinates and pixel indices
    coords = [dataset.xy(row, col) for row, col in high_confidence_indices]
    pixels = [tuple(idx) for idx in high_confidence_indices]
    
    return coords, pixels


def crop(input_path: str, bbox_coords: tuple[float, float, float, float], output_dir: str = "../data/modis/processing") -> str:
    """
    Saves a cropped version of a MODIS image based on bounding box coordinates.

    Parameters:
    input_path (str): The path to the input MODIS image.
    bbox_coords (tuple[float, float, float, float]): Bounding box coordinates (minx, miny, maxx, maxy).
    output_dir (str, optional): The directory to save the cropped image. Defaults to "../data/modis".

    Returns:
    str: The full path to the cropped image.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the MODIS image
    with rasterio.open(input_path) as src:
        # Get the window corresponding to the bounding box
        minx, miny, maxx, maxy = bbox_coords
        window = src.window(minx, miny, maxx, maxy)

        # Read the data from the window
        modis_data = src.read(window=window)

        # Get the profile of the cropped data
        profile = src.profile.copy()
        profile.update(
            {
                "height": window.height,
                "width": window.width,
                "transform": src.window_transform(window),
            }
        )

        # Save the new file to tiff
        output_filename = os.path.basename(input_path).replace('.tif', '_cropped.tif')
        output_full_path = os.path.join(output_dir, output_filename)

        with rasterio.open(output_full_path, "w", **profile) as dst:
            dst.write(modis_data)

    return output_full_path
    
    
def resize(input_path: str, output_dir: str = "../data/modis/processing") -> str:
    """
    Resizes a clipped MODIS image to a new width and height.

    Parameters:
    input_path (str): The path to the input clipped MODIS image.
    output_dir (str): The directory to save the resized image.

    Returns:
    str: The full path to the resized MODIS image.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the clipped MODIS image
    with rasterio.open(input_path) as src:
        # Define new width and height
        new_width, new_height = 128, 128

        # Resize the image
        clipped_modis_resized = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.bilinear
        )

        # Get the profile of the resized MODIS data
        profile = src.profile.copy()
        profile.update({
            "height": new_height,
            "width": new_width,
            "transform": src.transform
        })

        # Save the new file to tiff
        output_filename = os.path.basename(input_path).replace('.tif', '_resized.tif')
        output_full_path = os.path.join(output_dir, output_filename)

        # Write the resized MODIS data to a new GeoTIFF file
        with rasterio.open(output_full_path, "w", **profile) as dst:
            dst.write(clipped_modis_resized)

    return output_full_path
