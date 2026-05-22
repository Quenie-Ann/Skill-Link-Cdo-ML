# Skill-Link-Cdo-ML / matcher/schema.py
# Pydantic request and response models for the POST /match/ endpoint.
# FastAPI uses these for automatic payload validation and OpenAPI documentation.

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
 
 
class JobRequestPayload(BaseModel):
    """
    Slim representation of the job request forwarded by the Django API.
 
    job_description:
        Built from the selected JobType.description plus the optional
        JobRequest.description (resident note).
 
    budget_min / budget_max:
        Soft scoring signals. Both nullable — if neither is provided,
        the price signal returns a neutral 0.5 score.
 
    location_lat / location_lng:
        Required for the Haversine proximity calculation.
    """
    job_description: str = Field(
        ...,
        description=(
            "Canonical job type description (Admin-defined) with optional "
            "resident note appended. Primary TF-IDF input."
        ),
    )
    budget_min: Optional[float] = Field(default=None, ge=0)
    budget_max: Optional[float] = Field(default=None, ge=0)
    location_lat: float = Field(..., ge=-90, le=90)
    location_lng: float = Field(..., ge=-180, le=180)
 
 
class CandidateWorker(BaseModel):
    """
    Lightweight worker record. Only the fields the ML engine needs.
    Full profile data is fetched from PostgreSQL by Django after ranking.
    """
    worker_id: str
    bio: Optional[str] = Field(default=None)
    declared_rate: float = Field(..., ge=0)
    avg_rating: float = Field(default=0.0, ge=0.0, le=5.0)
    address_lat: Optional[float] = Field(default=None, ge=-90, le=90)
    address_lng: Optional[float] = Field(default=None, ge=-180, le=180)
 
 
class MatchRequest(BaseModel):
    """Full payload sent by the Django API to POST /match/."""
    job_request: JobRequestPayload
    candidates: list[CandidateWorker]
 
 
class RankedWorker(BaseModel):
    worker_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    score_breakdown: dict = Field(default_factory=dict)
 
 
class MatchResponse(BaseModel):
    """Ranked worker list returned to the Django API."""
    ranked: list[RankedWorker]
 