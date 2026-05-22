# Skill-Link-Cdo-ML / main.py
# FastAPI ML Matching Service entry point.

import os
import logging
from fastapi import FastAPI, Request, HTTPException

from dotenv import load_dotenv

from matcher.schema import MatchRequest, MatchResponse
from matcher.engine import run_matching

load_dotenv()

SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(name)s: %(message)s",
)
logger = logging.getLogger("skilllink-ml")

app = FastAPI(
    title="Skill-Link CDO — ML Matching Service",
    description=(
        "Standalone FastAPI microservice. Scores and ranks pre-filtered "
        "worker candidates for a given job request. Called exclusively by "
        "the Django REST API via an authenticated HTTP POST."
    ),
    version="1.0.0",
)

# No CORS configured — this service is internal only.

def _verify_key(request: Request) -> None:
    """
    Validates X-Service-Key header. Rejects with HTTP 403 if missing or wrong.
    Called at the top of every /match/ request. (SRS Section 12.3)
    """
    if not SERVICE_API_KEY:
        logger.critical("SERVICE_API_KEY is not set. All requests will be rejected.")
        raise HTTPException(status_code=403, detail="Forbidden")
    incoming = request.headers.get("X-Service-Key", "")
    if incoming != SERVICE_API_KEY:
        logger.warning(
            "Rejected request — invalid X-Service-Key from %s",
            request.client.host if request.client else "unknown",
        )
        raise HTTPException(status_code=403, detail="Forbidden")


@app.get("/health/", tags=["Health"])
def health_check():
    """Health check used by Render and Django startup validation. (SRS Section 7.3)"""
    return {"status": "ok"}


@app.post("/match/", response_model=MatchResponse, tags=["Matching"])
def match_workers(payload: MatchRequest, request: Request):
    """
    Receives a pre-filtered candidate list from the Django API and returns
    a ranked list of worker IDs with composite scores.

    The Django API has already applied two hard filters before calling here:
      1. verification_status == 'verified'
      2. skill_category == the category selected by the resident from the UI
    This service applies NO category logic.
    """
    _verify_key(request)

    if not payload.candidates:
        logger.info("Received match request with zero candidates.")
        return MatchResponse(ranked=[])

    logger.info(
        "Match request — candidates: %d",
        len(payload.candidates),
    )

    ranked = run_matching(payload.job_request, payload.candidates)

    logger.info(
        "Ranked %d workers. Top score: %.4f",
        len(ranked),
        ranked[0].score if ranked else 0,
    )
    return MatchResponse(ranked=ranked)