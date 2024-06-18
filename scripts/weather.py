import datetime
import os

from meteostat import Daily, Point
import numpy as np
import rasterio
import rioxarray as rxr
import tqdm
from scipy.interpolate import griddata


def get_weather(
    input_path: str,
    date_start: datetime.date,
    date_end: datetime.date,
    output_dir: str = "../data/modis/final",
    mask_types=["tavg", "prcp", "wspd", "sin_wdir", "cos_wdir"],
    interpolate=True
):
    """
    Main function to get weather data, reshape it into masks, and save the masks to files.

    Parameters:
    - input_path: Path to the input MODIS raster file.
    - date_start: Start date for weather data fetching.
    - date_end: End date for weather data fetching.
    - output_dir: Directory to save the output mask files.
    - mask_types: Must be str in "'tavg','tmin','tmax','prcp','snow','wdir','wspd','wpgt','pres','tsun'". List of weather features to include in the masks.
    - interpolate: provides interpolation on fetched data, using griddata. Is False by default, can be set to "True".
    
    Returns:
    - files: list of path to weather data
    """
    file_paths = []
    for mask in mask_types:
        # Create the filename based on the mask key
        filename = f"{os.path.basename(input_path)}.{mask}"
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            file_paths.append(path)
        else:
            data = fetch_weather(input_path, date_start, date_end)
            masks = reshape_weather(input_path, data, mask_types)
            if(interpolate):
                masks = interpolate_weather(masks)
            file_paths = save_weather(masks, input_path, output_dir)
            break
    return file_paths


def fetch_weather(input_path: str, date_start: datetime.date, date_end: datetime.date):
    """
    Fetch weather data for all points in the raster file within the specified date range.

    Parameters:
    - input_path: Path to the input MODIS raster file.
    - date_start: Start date for weather data fetching.
    - date_end: End date for weather data fetching.

    Returns:
    - all_data: List of DataFrames containing weather data for each point.
    """
    # Open the input raster dataset
    dataset = rasterio.open(input_path)

    # Generate the coordinates for all points
    cols, rows = np.meshgrid(np.arange(dataset.width), np.arange(dataset.height))
    xs, ys = rasterio.transform.xy(dataset.transform, rows, cols)

    # Flatten the arrays and convert to points
    flat_xs, flat_ys = np.array(xs).flatten(), np.array(ys).flatten()
    points = [Point(y, x) for x, y in zip(flat_xs, flat_ys)]

    # Fetch data for all points and dates
    all_data = []
    for id, point in enumerate(tqdm.tqdm(points)):
        data = Daily(point, date_start, date_end).fetch()
        if not data.empty:
            data["point_id"] = id
            all_data.append(data)

    return all_data


def reshape_weather(
    input_path: str,
    weather_data: list,
    mask_types: list[str] = ["tavg", "prcp", "wspd", "sin_wdir", "cos_wdir"],
):
    """
    Reshape fetched weather data into masks of shape (time, x, y).

    Parameters:
    - input_path: Path to the input MODIS raster file.
    - weather_data: List of DataFrames containing weather data for each point.
    - mask_types: List of weather features to include in the masks.

    Returns:
    - transposed_masks: Dictionary of reshaped masks for each weather feature.
    """
    tiff_file = rxr.open_rasterio(input_path)

    # Initialize the dictionary to store masks for each day
    masks = {
        mask_type: np.full(
            (tiff_file.shape[0], tiff_file.shape[1], tiff_file.shape[2]),
            np.nan,
            dtype=float,
        )
        for mask_type in mask_types
    }

    # Populate masks with weather data
    for df in weather_data:
        # Find localization of data point in base array
        id = df["point_id"].iloc[0]
        x = int(id / 128)
        y = id % 128

        # Use circular encoding for wdir
        wdir = np.radians(df["wdir"].to_numpy())
        for mask in masks:
            if mask == "sin_wdir":
                data = np.sin(wdir)
            elif mask == "cos_wdir":
                data = np.cos(wdir)
            else:
                data = df[mask].to_numpy()
            masks[mask][:, x, y] = data

    # Transpose to (time, x, y)
    transposed_masks = {
        mask_type: np.transpose(mask, (0, 2, 1)) for mask_type, mask in masks.items()
    }

    return transposed_masks


def interpolate_weather(masks):
    # Interpolate missing data using nearest neighbor interpolation
    for mask_type, mask in masks.items():
        for t in range(mask.shape[0]):
            try:
                x, y = np.indices(mask[t].shape)
                valid_mask = ~np.isnan(mask[t])
                points = np.column_stack((x[valid_mask], y[valid_mask]))
                values = mask[t][valid_mask]
                
                grid_x, grid_y = np.indices(mask[t].shape)
                mask_interpolated = griddata(
                    points, values, (grid_x, grid_y), method='linear'
                )
                
                # Ensure mask_interpolated has the correct shape (128, 128)
                mask_interpolated = np.nan_to_num(mask_interpolated, nan=np.nan)
                
                # Assign interpolated mask to transposed_masks
                masks[mask_type][t] = mask_interpolated
            except ValueError:
                print("No linear interpolation: no data")

    for mask_type, mask in masks.items():
        for t in range(mask.shape[0]):
            try: 
                x, y = np.indices(mask[t].shape)
                valid_mask = ~np.isnan(mask[t])
                points = np.column_stack((x[valid_mask], y[valid_mask]))
                values = mask[t][valid_mask]

                grid_x, grid_y = np.indices(mask[t].shape)
                mask_interpolated = griddata(
                    points, values, (grid_x, grid_y), method='nearest'
                )
                
                # Ensure mask_interpolated has the correct shape (128, 128)
                mask_interpolated = np.nan_to_num(mask_interpolated, nan=np.nan)
                
                # Assign interpolated mask to transposed_masks
                masks[mask_type][t] = mask_interpolated
            except ValueError:
                print("No nearest interpolation: no data")

        return masks

def save_weather(
    data: dict, input_path, output_dir: str = "../data/modis/final"
) -> list[str]:
    """
    Save reshaped weather masks to TIFF files.

    Parameters:
    - data: Dictionary of reshaped masks for each weather feature.
    - input_path: Path to the input MODIS raster file.
    - output_dir: Directory to save the output mask files.

    Returns:
    - list[str]: list of path to which the weather data was saved
    """
    # Open the input raster dataset
    dataset = rasterio.open(input_path)
    profile = dataset.profile.copy()

    profile.update(
        {
            "dtype": "float64",
        }
    )

    # Get the list of saved files
    files = []
    for key in data.keys():
        # Create the filename based on the mask key
        filename = f"{os.path.basename(input_path)}.{key}"
        path = os.path.join(output_dir, filename)
        files.append(path)
        # save it
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data[key])
    return files
