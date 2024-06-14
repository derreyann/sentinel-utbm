from pydantic import BaseModel, confloat, field_validator, ValidationInfo
import datetime
from dataclasses import dataclass
from typing import Union, Tuple, Annotated
from annotated_types import Len

GPSCoord = Tuple[confloat(ge=-90, le=90), confloat(ge=-180, le=180)]
BoundingBox = Annotated[list[GPSCoord], Len(4)]

@dataclass
class Event(BaseModel):
    start_date: datetime.date
    end_date: datetime.date
    location: Union[GPSCoord, BoundingBox]

    @field_validator("end_date")
    def check_date_order(cls, v: datetime.date, info: ValidationInfo) -> datetime.date:
        if "start_date" in info.data and v < info.data["start_date"]:
            raise ValueError('End date must be after start date.')
        return v
    
    @field_validator('location')
    def check_location(cls, location):
        if isinstance(location, tuple) and len(location) == 2:
            return location
        elif isinstance(location, list) and len(location) == 4:
            return location
        else:
            raise ValueError("Invalid location. Must be a GPS coordinate (latitude, longitude) or a bounding box (list of 4 GPS coordinates).")
        
    def __init__(self, start_date: datetime.date, end_date: datetime.date, location: Union[GPSCoord, BoundingBox]):
        super().__init__(start_date = start_date, end_date = end_date, location = location)


