from datetime import datetime, timedelta
import os
import warnings

import numpy as np
from meteostat import Daily, Point
import pandas as pd
import pymodis
import rasterio
from rasterio.warp import Resampling
import rioxarray as rxr
import tqdm
import yaml
import xarray as xr
import math


def dataflow(
    start_date,
    end_date,
    tiles: str = "h14v03",
    product: str = "MOD14A1.061",
    raw_dir: str = "../data/modis/raw",
    processing_dir: str = "../data/modis/processing",
    output_dir: str = "../data/modis/final",
):
    # Download data
    textfile_path = download_modis(
        start_date, end_date, output_dir=raw_dir, tiles=tiles, product=product
    )

    # Get all file names
    textfile_name = f"listfile{product}.txt"
    raw_dir = os.path.join(raw_dir, tiles)
    textfile_path = os.path.join(raw_dir, textfile_name)
    hdf_files = get_modis_hdf_filelist(textfile_path)

    fire_list = []
    for file in hdf_files:
        fire_list.append(extract_fire_mask(file, processing_dir))
    return fire_list


def get_modis_hdf_filelist(textfile_path: str):
    hdf_files = []
    dir = os.path.dirname(textfile_path)
    with open(textfile_path) as file:
        for line in file:
            line = line.strip()
            if line.endswith(".hdf"):
                hdf_files.append(os.path.join(dir, line))
    return hdf_files


def download_modis(
    start_date,
    end_date,
    output_dir: str = "../data/modis/raw",
    tiles: str = "h14v03",
    path: str = "MOLT",
    product: str = "MOD14A1.061",
):
    """Fetches all the modis files between specific dates"""
    full_path = os.path.join(output_dir, tiles)

    textfile_name = f"listfile{product}.txt"
    final_output_dir = os.path.join(output_dir, tiles)
    textfile_path = os.path.join(final_output_dir, textfile_name)
    if os.path.exists(textfile_path):
        return textfile_path

    # Create directory for our data
    os.mkdir(full_path)

    # Fetch credentials for modis
    with open("../config.yaml") as file:
        credentials = yaml.safe_load(file)
    user = credentials["modis"]["API_USER"]
    password = credentials["modis"]["API_PASSWORD"]

    # Setup API call
    downloader = pymodis.downmodis.downModis(
        destinationFolder=full_path,
        user=user,
        password=password,
        today=start_date,
        enddate=end_date,
        product=product,
        path=path,
        tiles=tiles,
    )

    # Call
    downloader.connect()
    # Fetch credentials for modis
    with open("../config.yaml") as file:
        credentials = yaml.safe_load(file)
    user = credentials["modis"]["API_USER"]
    password = credentials["modis"]["API_PASSWORD"]
    downloader.downloadsAllDay(allDays=False)
    return textfile_path


def extract_fire_mask(
    input_path: str, output_dir: str = "../data/modis/processing"
) -> tuple[xr.Dataset, list[datetime], str]:
    """
    Extracts the fire mask from a MODIS HDF file, reprojects it to EPSG:4326,
    and saves it as a GeoTIFF file. Also extracts the dates from the file attributes.

    Parameters:
    input_path (str): The path to the input HDF file.
    output_dir (str, optional): The directory to save the output file. If None, saves in input_dir.

    Returns:
    tuple[xr.Dataset, list[datetime], str]: The reprojected fire mask dataset, list of dates and the full path to the processed file.
    """
    # check if the file already exists
    output_filename = os.path.basename(input_path)[:-3] + "fire.tif"
    output_full_path = os.path.join(output_dir, output_filename)
    if os.path.exists(output_full_path):
        return output_full_path

    # Retrieve data
    dataset = rxr.open_rasterio(input_path, masked=True)

    # Reproject
    dataset = dataset.rio.reproject("EPSG:4326")

    # TODO: remove if not needed with new weather functions
    # Extracts dates
    # dates = dataset.attrs["DAYSOFYEAR"]
    # dates = dates.split(", ")
    # date_objects = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in dates]

    dataset.FireMask.rio.write_nodata(0, inplace=True)
    dataset = dataset.FireMask.rio.reproject("EPSG:4326")

    dataset.rio.to_raster(output_full_path)
    return output_full_path


def get_coords_and_pixels(
    dataset: rasterio.io.DatasetReader,
) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
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


def crop(
    input_path: str,
    bbox_coords: tuple[float, float, float, float],
    output_dir: str = "../data/modis/processing",
) -> str:
    """
    Saves a cropped version of a MODIS image based on bounding box coordinates.

    Parameters:
    input_path (str): The path to the input MODIS image.if not os.path.exists():
    bbox_coords (tuple[float, float, float, float]): Bounding box coordinates (minx, miny, maxx, maxy).
    output_dir (str, optional): The directory to save the cropped image. Defaults to "../data/modis".

    Returns:
    str: The full path to the cropped image.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the new file to tiff
    output_filename = os.path.basename(input_path).replace(".tif", ".cropped.tif")
    output_full_path = os.path.join(output_dir, output_filename)
    if os.path.exists(output_full_path):
        return output_full_path

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

        with rasterio.open(output_full_path, "w", **profile) as dst:
            dst.write(modis_data)

    return output_full_path


def resize(input_path: str, output_dir: str = "../data/modis/final") -> str:
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

    # Save the new file to tiff
    output_filename = os.path.basename(input_path).replace(".tif", ".cropped.tif")
    output_full_path = os.path.join(output_dir, output_filename)
    if os.path.exists(output_full_path):
        return output_full_path

    # Open the clipped MODIS image
    with rasterio.open(input_path) as src:
        # Define new width and height
        new_width, new_height = 128, 128

        # Resize the image
        clipped_modis_resized = src.read(
            out_shape=(src.count, new_height, new_width), resampling=Resampling.bilinear
        )

        # Get the profile of the resized MODIS data
        profile = src.profile.copy()
        profile.update(
            {"height": new_height, "width": new_width, "transform": src.transform}
        )

        # Write the resized MODIS data to a new GeoTIFF file
        with rasterio.open(output_full_path, "w", **profile) as dst:
            dst.write(clipped_modis_resized)

    return output_full_path


def get_tile(
    lat_geographic: float, lon_geographic: float
) -> tuple[int, int, float, float]:
    """
    Get the tile index and pixel coordinates of a given geographic coordinate.

    Parameters:
    lat_geographic (float): The latitude of the geographic coordinate.
    lon_geographic (float): The longitude of the geographic coordinate.

    Returns:
    tuple[int, int, float, float]: A tuple containing the vertical tile index, horizontal tile index, line and sample pixel coordinates.
    """

    if lat_geographic < -90 or lat_geographic > 90:
        raise ValueError("lat_geographic should be in range of [-90, 90]")
    if lon_geographic < -180 or lon_geographic > 180:
        raise ValueError("lon_geographic should be in range of [-180, 180]")

    lat_tile = lat_geographic
    lon_tile = lon_geographic * math.cos(math.radians(lat_geographic))

    vertical_tile = int((90 - lat_tile) / 10)
    horizontal_tile = int((lon_tile + 180) / 10)
    line = -(lat_tile - 90 + vertical_tile * 10) * 120 - 0.5
    sample = (180 - horizontal_tile * 10 + lon_tile) * 120 - 0.5
    tile = f"h{0:02}v{1:02}".format(vertical_tile, horizontal_tile)
    return tile


def get_weather(input_path: str, date_objects):
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Open the resized MODIS image
    tiff_file = rxr.open_rasterio(input_path)

    # Initialize the dictionary to store masks
    mask_types = ["tavg", "prcp", "wspd", "sin_wdir", "cos_wdir"]
    masks = {
        mask_type: np.zeros_like(tiff_file, dtype=float) for mask_type in mask_types
    }

    # Open the input raster dataset
    dataset = rasterio.open(input_path)
    profile = dataset.profile.copy()

    # TODO: Vectorize operations
    for k, date in enumerate(date_objects):
        print(f"Processing data for date: {date}")

        for i in tqdm.tqdm(range(dataset.read(k + 1).shape[0])):
            for j in range(dataset.read(k + 1).shape[1]):
                point = Point(j, i)

                # Get daily data
                data = Daily(point, date, date)
                data = data.interpolate()
                data = data.fetch()

                # If data is not empty, assign values to the mask arrays
                if not data.empty:
                    for mask_type in mask_types:
                        cossin_data = np.radians(data)
                        # add circular encoding to wind direction
                        if mask_type == "sin_wdir":
                            masks["sin_wdir"][k, i, j] = np.sin(cossin_data)
                        if mask_type == "cos_wdir":
                            masks["cos_wdir"][k, i, j] = np.cos(cossin_data)

                        masks[mask_type][k, i, j] = data[mask_type]
                else:
                    for mask_type in mask_types:
                        masks[mask_type][k, i, j] = np.nan

    save_weather(masks, profile)
    return masks


def save_weather(data: dict, profile, output_dir: str = "../data/modis/final"):
    profile.update(
        {
            "dtype": "float64",
        }
    )
    for key in data.keys():
        path = os.path.join(output_dir, key)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data[key])
