import os
import shutil
import glob
from osgeo import gdal
import cv2
import matplotlib.pyplot as plt
from geopy.distance import distance
from geopy.point import Point
from sentinelhub import SentinelHubRequest, DataCollection, MimeType
import math
import numpy as np

from sentinelhub import (
    CRS,
    BBox,
    bbox_to_dimensions,
)


def move_coords(longitude, latitude, dist, dir):
    """
    Move the coordinates by a given distance in a given direction.

    Parameters:
    longitude (float): The initial longitude.
    latitude (float): The initial latitude.
    dist (float): The distance to move in kilometers.
    dir (float): The direction in degrees.

    Returns:
    tuple: The new latitude and longitude.
    """

    initial_point = Point(latitude, longitude)
    new_point = distance(kilometers=dist).destination(initial_point, dir)
    new_latitude = new_point.latitude
    new_longitude = new_point.longitude
    return new_latitude, new_longitude


def generate_grid_within_box(lat_min, lon_min, lat_max, lon_max, spacing_km):
    """
    Generate a grid of points within a given bounding box.

    Parameters:
    lat_min (float): The minimum latitude of the bounding box.
    lon_min (float): The minimum longitude of the bounding box.
    lat_max (float): The maximum latitude of the bounding box.
    lon_max (float): The maximum longitude of the bounding box.
    spacing_km (float): The spacing between points in kilometers.

    Returns:
    tuple: A tuple containing a list of points, the maximum number of points in a line, the number of lines, and a list of the number of points per line.
    """

    points = []
    num_lines = 0
    max_points_in_line = 0
    points_per_line = []

    # Initial coordinates
    current_lat = lat_max
    current_lon = lon_min

    while current_lat >= lat_min:
        num_lines += 1
        points_in_line = 0
        line_points = []

        while current_lon <= lon_max:
            # Check if the next point will be outside the bbox
            next_lat, next_lon = move_coords(current_lon, current_lat, spacing_km, 90)
            line_points.append((current_lat, current_lon))
            points_in_line += 1
            current_lat, current_lon = next_lat, next_lon

        # Check if the next point will be outside the bbox
        next_lat, next_lon = move_coords(current_lon, current_lat, spacing_km, 90)
        line_points.append((current_lat, current_lon))
        points_in_line += 1
        current_lat, current_lon = next_lat, next_lon

        points.extend(line_points)
        points_per_line.append(points_in_line)
        max_points_in_line = max(max_points_in_line, points_in_line)

        # Move to the next line, shifting current_lon to make the next line start further away
        current_lat, current_lon = move_coords(
            lon_min, current_lat, spacing_km - 10, 180
        )

    return points, max_points_in_line, num_lines, points_per_line


# degrees to radians
def deg2rad(degrees):
    return math.pi * degrees / 180.0


# radians to degrees
def rad2deg(radians):
    return 180.0 * radians / math.pi


# Semi-axes of WGS-84 geoidal reference
WGS84_a = 6378137.0  # Major semiaxis [m]
WGS84_b = 6356752.3  # Minor semiaxis [m]


# Earth radius at a given latitude, according to the WGS-84 ellipsoid [m]
def WGS84EarthRadius(lat):
    # http://en.wikipedia.org/wiki/Earth_radius
    An = WGS84_a * WGS84_a * math.cos(lat)
    Bn = WGS84_b * WGS84_b * math.sin(lat)
    Ad = WGS84_a * math.cos(lat)
    Bd = WGS84_b * math.sin(lat)
    return math.sqrt((An * An + Bn * Bn) / (Ad * Ad + Bd * Bd))


# Bounding box surrounding the point at given coordinates,
# assuming local approximation of Earth surface as a sphere
# of radius given by WGS84
def boundingBox(longitudeInDegrees, latitudeInDegrees, halfSideInKm):
    """
    Compute the bounding box around a point on the Earth, given a distance from the point.

    Parameters:
    longitudeInDegrees (float): The longitude of the point.
    latitudeInDegrees (float): The latitude of the point.
    halfSideInKm (float): Half the side of the bounding box in kilometers.

    Returns:
    tuple: A tuple containing the minimum longitude, minimum latitude, maximum longitude, and maximum latitude of the bounding box.
    """

    lat = deg2rad(latitudeInDegrees)
    lon = deg2rad(longitudeInDegrees)
    halfSide = 1000 * halfSideInKm

    # Radius of Earth at given latitude
    radius = WGS84EarthRadius(lat)
    # Radius of the parallel at given latitude
    pradius = radius * math.cos(lat)

    latMin = lat - halfSide / radius
    latMax = lat + halfSide / radius
    lonMin = lon - halfSide / pradius
    lonMax = lon + halfSide / pradius

    return (rad2deg(lonMin), rad2deg(latMin), rad2deg(lonMax), rad2deg(latMax))


# fonction qui configure la requete sentinel (request_ndvi_img) et get_data (ndvi_img):


def get_ndvi_img(
    aoi_bbox, aoi_size, data_folder, start_date, end_date, evalscript_ndvi, config
):
    """
    Gather sentinel data with a given evalscript and configuration.

    Parameters:
    aoi_bbox (tuple): The bounding box of the area of interest.
    aoi_size (tuple): The size of the area of interest.
    data_folder (str): The folder to save the data.
    start_date (str): The start date of the data.
    end_date (str): The end date of the data.
    evalscript_ndvi (str): The evalscript to use to calculate the NDVI.
    config (sh.SentinelHubConfig): The configuration to use for the request.

    Returns:
    numpy.ndarray: The image and .tiff file in the data folder.
    """

    request_ndvi_img = SentinelHubRequest(
        data_folder=data_folder,
        evalscript=evalscript_ndvi,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(start_date, end_date),
                other_args={"dataFilter": {"mosaickingOrder": "leastCC"}},
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=aoi_bbox,
        size=aoi_size,
        config=config,
    )

    ndvi_img = request_ndvi_img.get_data(save_data=True)
    return ndvi_img


def concatenate_tiff_images(
    sentinel_request_dir, sentinel_tiff_dir, sentinel_merged_dir
):
    """
    Concatenate the TIFF images in the given directory into a single TIFF image.

    Parameters:
    sentinel_request_dir (str): The directory containing the TIFF images.
    sentinel_tiff_dir (str): The directory to save the TIFF images.
    sentinel_merged_dir (str): The directory to save the merged TIFF image.

    Returns:
    Save the merged TIFF image in the sentinel_merged_dir and show the image.
    """
    
    # Delete all XML files in the directory
    for file in glob.glob(sentinel_tiff_dir + "/**/*.tiff", recursive=True):
        os.remove(file)

    # Gather the paths of the .tiff files
    tiff_files = glob.glob(sentinel_request_dir + "/**/*.tiff", recursive=True)
    
    if len(tiff_files) == 0:
        raise ValueError("No .tiff files found in the directory.")
    
    print(tiff_files)

    if(not os.path.exists(sentinel_tiff_dir)):
        os.mkdir(sentinel_tiff_dir)
        
    # Move the .tiff images to the directory and increment the name
    for i, tiff_file in enumerate(tiff_files):
        shutil.move(tiff_file, sentinel_tiff_dir + f"/image_{i}.tiff")

    # Load all TIFF images
    tiff_images = [
        cv2.imread(tiff_file, cv2.IMREAD_UNCHANGED) for tiff_file in tiff_files
    ]

    # Delete all XML files in the directory
    for file in glob.glob(sentinel_tiff_dir + "/**/*.xml", recursive=True):
        os.remove(file)
    # delete all files in the directory request
    shutil.rmtree(sentinel_request_dir)

    # Get the paths of .tiff files
    tiff_files = glob.glob(sentinel_tiff_dir + "/*.tiff")

    # Open all TIFF files
    tiffs = [gdal.Open(tiff_file) for tiff_file in tiff_files]

    # Get the metadata of the first image
    metadata = tiffs[0].GetMetadata()
    geotransform = tiffs[0].GetGeoTransform()
    projection = tiffs[0].GetProjection()
    
    print("--" * 10 + "INFO" + "--" * 10)
    print("Metadata:", metadata)
    print("Geotransform:", geotransform)
    print("Projection:", projection)
    print("--" * 10 + "INFO" + "--" * 10)
    
    if(not os.path.exists(sentinel_merged_dir)):
        os.mkdir(sentinel_merged_dir)

    # Initial path of the output file
    output_file = os.path.join(sentinel_merged_dir, "merged_image.tiff")
    vrt_output_file =  os.path.join(sentinel_merged_dir, "merged.vrt")

    # Increment the filename if it already exists
    base, extension = os.path.splitext(output_file)
    vrt_base, extension_vrt = os.path.splitext(output_file)
    counter = 1
    while os.path.exists(output_file):
        output_file = f"{base}_{counter}{extension}"
        vrt_output_file = f"{vrt_base}_{counter}{extension_vrt}"
        counter += 1

    # Create a VRT (Virtual Dataset) from the TIFF files
    vrt_options = gdal.BuildVRTOptions(resampleAlg="nearest")
    vrt = gdal.BuildVRT(vrt_output_file, tiff_files, options=vrt_options)
    # Convert the VRT to a TIFF
    gdal.Translate(output_file, vrt)

    return output_file


def create_stitched_image(lat_min, lon_min, lat_max, lon_max, spacing_km, resolution, start_date, end_date, evalscript_ndvi, config, sentinel_request_dir, sentinel_tiff_dir, sentinel_merge_dir):
    points, max_points_in_line, num_lines, points_per_line = generate_grid_within_box(lat_min, lon_min, lat_max, lon_max, spacing_km)

    points = [(y, x) for x, y in points]

    for point in points:
        bbox = boundingBox(point[0],point[1], spacing_km / 2)
        aoi_bbox_list = (BBox(bbox=bbox, crs=CRS.WGS84))
        aoi_size = bbox_to_dimensions(aoi_bbox_list, resolution=resolution)
        get_ndvi_img(aoi_bbox_list, aoi_size, sentinel_request_dir, start_date, end_date, evalscript_ndvi, config)

    output_file = concatenate_tiff_images(sentinel_request_dir, sentinel_tiff_dir, sentinel_merge_dir)
    return output_file