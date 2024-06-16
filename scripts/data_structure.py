import modis, weather, sentinel, evalscripts

from pydantic import BaseModel, confloat, field_validator, ValidationInfo
import datetime
from dataclasses import dataclass
import yaml
from sentinelhub import SHConfig


class Event:
    def __init__(self, start_date, end_date, latitude, longitude):
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

    def validate(self):
        # Perform the validation using Pydantic
        EventModel(
            start_date=self.start_date,
            end_date=self.end_date,
            latitude=self.latitude,
            longitude=self.longitude,
        )

    def get_modis_data(self):
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
    ):
        # need to add check to ensure modis files are available
        weather_files = weather.get_weather(
            self.modis_path[0], self.start_date, self.end_date, output_dir, masks
        )
        self.weather_path = weather_files

    def get_sentinel_data(
        self,
        spacing_km=100,
        resolution=300,
        evalscript=evalscripts.evalscript_ndvi,
        sentinel_request_dir="../data/sentinel/raw",
        sentinel_tiff_dir="../data/sentinel/processing",
        sentinel_merge_dir="../data/sentinel/final",
    ):
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
            start_date=self.start_date,
            end_date=self.end_date,
            evalscript_ndvi=evalscript,
            config=config,
            sentinel_request_dir=sentinel_request_dir,
            sentinel_tiff_dir=sentinel_tiff_dir,
            sentinel_merge_dir=sentinel_merge_dir,
        )
        self.sentinel_path = img_path


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
        latitude: confloat(ge=-90, le=90),
        longitude: confloat(ge=-180, le=180),
    ):
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            latitude=latitude,
            longitude=longitude,
        )

    @field_validator("end_date")
    def check_date_order(cls, v: datetime.date, info: ValidationInfo) -> datetime.date:
        """Validates that the end_date is not before the start_date"""
        if "start_date" in info.data and v < info.data["start_date"]:
            raise ValueError("End date must be after start date.")
        return v
