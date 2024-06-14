import datetime
import warnings

from meteostat import Daily, Point
import numpy as np
import rasterio
import rioxarray as rxr
import tqdm


def get_weather(input_path: str, date_start: datetime.date, date_end: datetime):
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
