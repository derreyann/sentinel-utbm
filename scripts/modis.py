from datetime import datetime
import math
import os

import numpy as np
import pymodis
from pyproj import Proj
import rasterio
from rasterio.warp import Resampling
import rioxarray as rxr
import yaml
import xarray as xr

from utils import create_bounding_box


def dataflow(
    start_date,
    end_date,
    lat: float,
    lon: float,
    product: str = "MOD14A1.061",
    raw_dir: str = "../data/modis/raw",
    processing_dir: str = "../data/modis/processing",
    output_dir: str = "../data/modis/final",
):
    tiles = get_tile(lat, lon)

    # Download data
    textfile_path = download_modis(
        start_date, end_date, output_dir=raw_dir, tiles=tiles, product=product
    )

    # Get all file names
    textfile_name = f"listfile{product}.txt"
    raw_dir = os.path.join(raw_dir, tiles)
    textfile_path = os.path.join(raw_dir, textfile_name)
    hdf_files = get_modis_hdf_filelist(textfile_path)

    fire_files, start_date, end_date, bbox_coords = get_event_dimensions(
        hdf_files, processing_dir
    )

    final_array = []
    for file in fire_files:
        cropped = crop(file, bbox_coords, processing_dir)
        final_array.append(resize(cropped, output_dir))
    return final_array, start_date, end_date, bbox_coords


def get_modis_hdf_filelist(textfile_path: str):
    hdf_files = []
    dir = os.path.dirname(textfile_path)
    with open(textfile_path) as file:
        for line in file:
            line = line.strip()
            if line.endswith(".hdf"):
                hdf_files.append(os.path.join(dir, line))
    return hdf_files


def get_event_dimensions(hdf_files, processing_dir):
    """This functions extracts the spatial and time dimension out of an event"""
    fire_files = []
    coordinates_array = []
    date_array = []
    # Extract coordinates out of the first detected fire event
    for file in hdf_files:
        fire_file, dates = extract_fire_mask(file, processing_dir)
        coords, _ = get_coords_and_pixels(fire_file)

        if not coords:
            # If we don't find fire pixel and have no previous events, continue
            if not coordinates_array:
                continue
            # If we already have coordinates and no new events, break
            else:
                break

        coordinates_array += coords
        date_array.append([dates[0], dates[-1]])
        fire_files.append(fire_file)

    if not fire_files:
        raise ValueError("No fire founds")

    # Extract the first and last date out of that array
    start_date = date_array[0][0]
    end_date = date_array[-1][1]
    bbox_coords = create_bounding_box(coordinates_array)
    return fire_files, start_date, end_date, bbox_coords


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
    os.makedirs(full_path, exist_ok=True)

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
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    dataset = rxr.open_rasterio(input_path, masked=True)
    # TODO: remove if not needed with new weather functions
    # Extracts dates
    dates = dataset.attrs["DAYSOFYEAR"]
    dates = dates.split(", ")
    date_objects = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in dates]

    # check if the file already exists
    output_filename = os.path.basename(input_path)[:-3] + "fire.tif"
    output_full_path = os.path.join(output_dir, output_filename)
    if os.path.exists(output_full_path):
        return output_full_path, date_objects

    # Retrieve data

    # Reproject
    dataset = dataset.rio.reproject("EPSG:4326")

    dataset.FireMask.rio.write_nodata(0, inplace=True)
    dataset = dataset.FireMask.rio.reproject("EPSG:4326")

    dataset.rio.to_raster(output_full_path)
    return output_full_path, date_objects


def get_coords_and_pixels(
    fire_filename: str,
) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    """
    Extracts the coordinates and pixels with high fire confidence from a fire band.

    Parameters:
    fire_filename: The input rasterio dataset with the Modis fire band.

    Returns:
    Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]: A tuple containing two lists:
        - List of coordinates (longitude, latitude) of high fire confidence pixels.
        - List of pixel indices (row, column) of high fire confidence pixels.
    """
    fire = rasterio.open(fire_filename)

    # Read the first band of the dataset
    fire_band = fire.read(1)

    # Get the indices of pixels with high fire confidence
    high_confidence_indices = np.argwhere(fire_band > 6)
    if not high_confidence_indices.any():
        return None, None

    # Extract the coordinates and pixel indices
    coords = [fire.xy(row, col) for row, col in high_confidence_indices]
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
    output_filename = os.path.basename(input_path).replace(".tif", ".resized.tif")
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
    CELLS = 2400
    VERTICAL_TILES = 18
    HORIZONTAL_TILES = 36
    EARTH_RADIUS = 6371007.181
    EARTH_WIDTH = 2 * math.pi * EARTH_RADIUS

    TILE_WIDTH = EARTH_WIDTH / HORIZONTAL_TILES
    TILE_HEIGHT = TILE_WIDTH

    MODIS_GRID = Proj(f"+proj=sinu +R={EARTH_RADIUS} +nadgrids=@null +wktext")

    x, y = MODIS_GRID(lon_geographic, lat_geographic)
    h = int((EARTH_WIDTH * 0.5 + x) / TILE_WIDTH)
    v = int(
        -(EARTH_WIDTH * 0.25 + y - (VERTICAL_TILES - 0) * TILE_HEIGHT) / TILE_HEIGHT
    )
    return f"h{h:02d}v{v:02d}"
