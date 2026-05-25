# Skill-Link-Cdo-ML / matcher/schema.py
# Pydantic request and response models for the POST /match/ endpoint.
# FastAPI uses these for automatic payload validation and OpenAPI documentation.

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class JobRequestSchema(BaseModel):
    job_type_name : str            = Field('', description="Admin-defined job type label (e.g. 'Fix leaking pipe')")
    description   : str            = Field('', description="Optional resident notes appended to the query")
    budget_min    : Optional[float] = None
    budget_max    : Optional[float] = None
    location_lat  : Optional[float] = None
    location_lng  : Optional[float] = None


class CandidateSchema(BaseModel):
    worker_id       : str
    declared_rate   : float
    avg_rating      : float   = 0.0
    years_experience: int     = 0
    address_lat     : Optional[float] = None
    address_lng     : Optional[float] = None
    bio             : Optional[str]   = None  # optional — empty bio gets neutral text score


class MatchRequestSchema(BaseModel):
    job_request : JobRequestSchema
    candidates  : list[CandidateSchema]


class RankedWorkerSchema(BaseModel):
    worker_id       : str
    score           : float
    score_breakdown : dict


class MatchResponseSchema(BaseModel):
    ranked : list[RankedWorkerSchema]