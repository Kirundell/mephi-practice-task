"""Pydantic-схемы запроса и ответа /predict."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class VisitRequest(BaseModel):
    """Один визит из ga_sessions."""

    session_id: str
    client_id: Optional[str] = None
    visit_date: str = Field(..., description="YYYY-MM-DD")
    visit_time: str = Field(..., description="HH:MM:SS")
    visit_number: int = 1

    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None
    utm_adcontent: Optional[str] = None
    utm_keyword: Optional[str] = None

    device_category: Optional[str] = None
    device_os: Optional[str] = None
    device_brand: Optional[str] = None
    device_model: Optional[str] = None
    device_screen_resolution: Optional[str] = None
    device_browser: Optional[str] = None

    geo_country: Optional[str] = None
    geo_city: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "9999999999.1234567890",
                "client_id": "1111111111.2222222222",
                "visit_date": "2022-03-15",
                "visit_time": "14:23:01",
                "visit_number": 2,
                "utm_source": "ZpYIoDJMcFzVoPFsHGJL",
                "utm_medium": "cpc",
                "utm_campaign": "LTuZkdKfxRGVceoWkVyg",
                "utm_adcontent": "vCIpmpaGBnIQhyYNkXqp",
                "utm_keyword": "puhZPIYqKXeFPaUviSjo",
                "device_category": "mobile",
                "device_os": "Android",
                "device_brand": "Samsung",
                "device_model": None,
                "device_screen_resolution": "412x915",
                "device_browser": "Chrome",
                "geo_country": "Russia",
                "geo_city": "Moscow",
            }
        }


class PredictionResponse(BaseModel):
    session_id: str
    prediction: int = Field(..., ge=0, le=1)
    probability: float = Field(..., ge=0.0, le=1.0)
    threshold: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
