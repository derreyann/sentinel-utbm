import modis

from pydantic import BaseModel, confloat, field_validator, ValidationInfo
import datetime
from dataclasses import dataclass
from typing import Tuple



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


    def validate(self):
        # Perform the validation using Pydantic
        EventModel(
            start_date=self.start_date,
            end_date=self.end_date,
            latitude=self.latitude,
            longitude=self.longitude,
        )


    def get_modis_data(self):
        final_array, start_date, end_date, bbox_coords = modis.dataflow(self.start_date, self.end_date, self.latitude, self.longitude)
        self.modis_path = final_array
        self.start_date = start_date
        self.end_date = end_date
        self.bbox_coords = bbox_coords

    def get_weather_data(self):
        return

    def get_sentinel_data(self):
        return


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
        super().__init__(start_date=start_date, end_date=end_date, latitude=latitude, longitude=longitude)
        
    @field_validator("end_date")
    def check_date_order(cls, v: datetime.date, info: ValidationInfo) -> datetime.date:
        """Validates that the end_date is not before the start_date"""
        if "start_date" in info.data and v < info.data["start_date"]:
            raise ValueError("End date must be after start date.")
        return v