# Skill-Link-Cdo-ML / main.py
# skilllink-ml / main.py
#
# FastAPI entry point. Exposes POST /match/ and GET /health/.
# ─────────────────────────────────────────────────────────────────────────────

import os
import logging
from fastapi import FastAPI, Header, HTTPException

from matcher.schema import MatchRequestSchema, MatchResponseSchema
from matcher.engine import compute_matches
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
SERVICE_API_KEY = os.getenv('SERVICE_API_KEY', '')

app = FastAPI(
    title='Skill-Link CDO — ML Matching Service',
    description='TF-IDF + weighted composite scoring for job-to-worker matching.',
    version='2.0.0',
)


def _verify_key(x_service_key: str):
    if not SERVICE_API_KEY:
        logger.warning('SERVICE_API_KEY not set — rejecting all requests.')
        raise HTTPException(status_code=403, detail='Forbidden')
    if x_service_key != SERVICE_API_KEY:
        raise HTTPException(status_code=403, detail='Forbidden')


@app.post('/match/', response_model=MatchResponseSchema)
async def match(
    payload: MatchRequestSchema,
    x_service_key: str = Header(...),
):
    _verify_key(x_service_key)
    ranked = compute_matches(payload.dict())
    return {'ranked': ranked}


@app.get('/health/')
async def health():
    return {'status': 'ok'}