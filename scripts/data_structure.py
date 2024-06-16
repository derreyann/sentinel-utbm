import modis, weather, sentinel, evalscripts

import os

from pydantic import BaseModel, confloat, field_validator, ValidationInfo
import datetime
from dataclasses import dataclass
import numpy as np
import rasterio
from sentinelhub import SHConfig
import yaml


class Event:
    def __init__(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        latitude: float,
        longitude: float,
    ):
        """
        Initialize the Event object with start date, end date, latitude, and longitude.

        Args:
        start_date (datetime.date): The start date of the event.
        end_date (datetime.date): The end date of the event.
        latitude (float): Latitude of the event location.
        longitude (float): Longitude of the event location.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.latitude = latitude
        self.longitude = longitude

        # Perform validation
        self.validate()

        # Initialize additional attributes
        self.modis_path = None
        self.bbox_coords = None
        self.weather_path = None
        self.sentinel_path = None

    def validate(self) -> None:
        """
        Validate the event attributes using Pydantic model.
        """
        # Perform the validation using Pydantic
        EventModel(
            start_date=self.start_date,
            end_date=self.end_date,
            latitude=self.latitude,
            longitude=self.longitude,
        )

    def get_modis_data(self) -> None:
        """
        Fetch MODIS data for the event and update the event attributes with the fetched data.
        """
        final_array, start_date, end_date, bbox_coords = modis.dataflow(
            self.start_date, self.end_date, self.latitude, self.longitude
        )
        self.modis_path = final_array
        self.start_date = start_date
        self.end_date = end_date
        self.bbox_coords = bbox_coords

    def get_weather_data(
        self,
        output_dir: str = "../data/modis/final",
        masks: list[str] = ["tavg", "prcp", "wspd", "sin_wdir", "cos_wdir"],
    ) -> None:
        """
        Fetch weather data for the event and update the event attributes with the fetched data.

        Args:
        output_dir (str): Directory to save the weather data.
        masks (list[str]): List of weather variables to fetch.
        """
        weather_path_list = []
        date = self.start_date
        for _ in self.modis_path:
            weather_files = weather.get_weather(
                self.modis_path[0],
                date,
                date + datetime.timedelta(days=7),
                output_dir,
                masks,
            )
            date += datetime.timedelta(days=8)
            weather_path_list.append(weather_files)
        self.weather_path = weather_path_list

    def get_sentinel_data(
        self,
        spacing_km: int = 100,
        resolution: int = 300,
        evalscript: str = evalscripts.evalscript_ndvi,
        sentinel_request_dir: str = "../data/sentinel/raw",
        sentinel_tiff_dir: str = "../data/sentinel/processing",
        sentinel_merge_dir: str = "../data/sentinel/final",
    ) -> None:
        """
        Fetch Sentinel data for the event and update the event attributes with the fetched data.

        Args:
        spacing_km (int): Spacing in kilometers for the Sentinel image.
        resolution (int): Resolution for the Sentinel image.
        evalscript (str): Evaluation script for Sentinel data processing.
        sentinel_request_dir (str): Directory to save the raw Sentinel requests.
        sentinel_tiff_dir (str): Directory to save the processed Sentinel TIFFs.
        sentinel_merge_dir (str): Directory to save the final merged Sentinel images.
        """
        # init storing arrays
        sentinel_path_list = []
        sentinel_resized_path_list = []
        date = self.start_date
        # Get the tile number to create the img path
        tile = modis.get_tile(self.latitude, self.longitude)
        # Loop over the number of modis files spanning 8 days
        for _ in self.modis_path:
            img_name = f"{tile}_{date.strftime("%Y-%m-%d")}_{(date + datetime.timedelta(days=7)).strftime("%Y-%m-%d")}"
            img_path = os.path.join(sentinel_merge_dir, f"{img_name}.tiff")
            if not os.path.exists(img_path):
                with open("../config.yaml") as file:
                    credentials = yaml.safe_load(file)
                user = credentials["sentinelhub"]["API_USER"]
                password = credentials["sentinelhub"]["API_PASSWORD"]
                config = SHConfig(sh_client_id=user, sh_client_secret=password)

                img_path = sentinel.create_stitched_image(
                    lat_min=self.bbox_coords[1],
                    lon_min=self.bbox_coords[0],
                    lat_max=self.bbox_coords[3],
                    lon_max=self.bbox_coords[2],
                    spacing_km=spacing_km,
                    resolution=resolution,
                    start_date=date,
                    end_date=date + datetime.timedelta(days=7),
                    evalscript_ndvi=evalscript,
                    config=config,
                    sentinel_request_dir=sentinel_request_dir,
                    sentinel_tiff_dir=sentinel_tiff_dir,
                    sentinel_merge_dir=sentinel_merge_dir,
                    img_name=img_name,
                )
            date += datetime.timedelta(days=8)
            resized_path = modis.resize(img_path, sentinel_merge_dir)
            sentinel_path_list.append(img_path)
            sentinel_resized_path_list.append(resized_path)
        self.sentinel_path = sentinel_path_list
        self.sentinel_resized_path = sentinel_resized_path_list

    def create_tensor_from_tiffs(self) -> np.ndarray:
        """
        Creates a tensor using the event's list of TIFF file paths.

        Returns:
        np.ndarray: Tensor containing stacked data from all TIFF files.
        """
        # Get the number of 8-day periods
        num_periods = len(self.modis_path)
        full_stack = []
        # Loop over each period
        for i in range(num_periods):
            weekly_stack = []

            # Dealing with daily data
            to_average_paths = []
            to_average_paths.append(self.modis_path[i])
            to_average_paths.extend(self.weather_path[i])

            for path in to_average_paths:
                with rasterio.open(path) as src:
                    array = src.read()  # Read all bands
                    array = np.mean(array, axis=0)
                    weekly_stack.append(array)

            # Dealing with Sentinel data
            if self.sentinel_path:
                with rasterio.open(self.sentinel_resized_path[i]) as src:
                    array = src.read()
                    for band in array:
                        weekly_stack.append(band)
            weekly_stack = np.stack(weekly_stack)
            full_stack.append(weekly_stack)
            shapes = [array.shape for array in weekly_stack]
        final_stack = np.stack(full_stack)
        return final_stack


@dataclass
class EventModel(BaseModel):
    start_date: datetime.date
    end_date: datetime.date
    latitude: confloat(ge=-90, le=90)
    longitude: confloat(ge=-180, le=180)

    def __init__(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        latitude: float,
        longitude: float,
    ):
        """
        Initialize the EventModel object with start date, end date, latitude, and longitude.

        Args:
        start_date (datetime.date): The start date of the event.
        end_date (datetime.date): The end date of the event.
        latitude (float): Latitude of the event location.
        longitude (float): Longitude of the event location.
        """
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            latitude=latitude,
            longitude=longitude,
        )

    @field_validator("end_date")
    def check_date_order(cls, v: datetime.date, info: ValidationInfo) -> datetime.date:
        """
        Validates that the end_date is not before the start_date.

        Args:
        v (datetime.date): The end date to validate.
        info (ValidationInfo): Additional information for validation.

        Returns:
        datetime.date: Validated end date.

        Raises:
        ValueError: If the end date is before the start date.
        """
        if "start_date" in info.data and v < info.data["start_date"]:
            raise ValueError("End date must be after start date.")
        return v
