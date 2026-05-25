# Skill-Link-Cdo-ML / matcher/engine.py
#
# Matching signals (v2 — updated per scope discussion):
#   1. Text relevance  — TF-IDF on job_type name + resident notes vs worker bio
#                        Bio is optional; empty bio gets neutral score (not excluded)
#   2. Proximity       — Haversine distance, exponential decay
#   3. Price           — budget range vs declared_rate, linear decay outside range
#   4. Rating          — avg_rating 0-5 normalized; new workers get neutral 0.5
#   5. Experience      — years_experience, square-root curve capped at config value
#
# Weights live in config.py — no code changes needed to re-tune.


from __future__ import annotations
import math
import logging
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import (
    WEIGHT_TEXT, WEIGHT_PROXIMITY, WEIGHT_PRICE,
    WEIGHT_RATING, WEIGHT_EXPERIENCE,
    PROXIMITY_DECAY_KM, EXPERIENCE_CAP_YEARS,
)

logger = logging.getLogger(__name__)


# ── Haversine ──────────────────────────────────────────────────────────────────

def _haversine_km(lat1, lng1, lat2, lng2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Signal scorers ─────────────────────────────────────────────────────────────

def _proximity_score(job_lat, job_lng, w_lat, w_lng):
    """Exponential decay: score = e^(-km / decay_km). Neutral 0.5 if no coords."""
    if w_lat is None or w_lng is None or job_lat is None or job_lng is None:
        return 0.5
    try:
        km = _haversine_km(float(job_lat), float(job_lng), float(w_lat), float(w_lng))
        return round(math.exp(-km / PROXIMITY_DECAY_KM), 4)
    except Exception:
        return 0.5


def _price_score(budget_min, budget_max, declared_rate):
    """
    1.0  inside range.
    Linear decay outside — how far the rate is outside expressed as a
    fraction of the range width. No budget supplied → neutral 0.5.
    """
    if budget_min is None or budget_max is None:
        return 0.5
    if budget_min <= declared_rate <= budget_max:
        return 1.0
    band = max(float(budget_max) - float(budget_min), 1.0)
    dist = (float(budget_min) - declared_rate
            if declared_rate < float(budget_min)
            else declared_rate - float(budget_max))
    return round(max(0.0, 1.0 - dist / band), 4)


def _rating_score(avg_rating):
    """
    Normalize 0-5 to 0-1.
    Workers with no ratings yet (0.0) receive 0.5 so they are not
    penalised before accumulating reviews.
    """
    if avg_rating <= 0:
        return 0.5
    return round(min(float(avg_rating) / 5.0, 1.0), 4)


def _experience_score(years):
    """
    Square-root curve capped at EXPERIENCE_CAP_YEARS.

    Early years carry strong weight; diminishing returns beyond the cap.
    This reflects real-world relevance — a 10-year worker is not twice
    as capable as a 5-year worker for a routine job.

    At cap = 10 years:
        0 yrs  → 0.00   5 yrs → 0.71
        1 yr   → 0.32  10 yrs → 1.00  (and beyond)
        3 yrs  → 0.55
    """
    if years <= 0:
        return 0.0
    capped = min(int(years), EXPERIENCE_CAP_YEARS)
    return round(math.sqrt(capped / EXPERIENCE_CAP_YEARS), 4)


# ── Text relevance ─────────────────────────────────────────────────────────────

def _build_query(job_type_name, resident_notes):
    """
    Primary query = admin-defined job type name (always present, always relevant).
    Resident notes appended when provided for additional context.
    This replaces the pure free-text description approach.
    """
    parts = [job_type_name.strip()] if job_type_name and job_type_name.strip() else []
    if resident_notes and resident_notes.strip():
        parts.append(resident_notes.strip())
    return ' '.join(parts)


def _text_scores(query, bios):
    """
    TF-IDF cosine similarity of query against each worker bio.
    Workers with empty/missing bio receive 0.4 (below average but not zero)
    so they appear in results but ranked below workers with relevant bios.
    This is the bio-optional fallback described in scope discussion.
    """
    n = len(bios)
    scores = [0.4] * n   # default for empty bios

    if not query.strip():
        return scores

    # Only vectorize workers who have a non-empty bio
    scored_idx  = [i for i, b in enumerate(bios) if b and b.strip()]
    if not scored_idx:
        return scores

    corpus = [bios[i].lower() for i in scored_idx]
    all_texts = [query.lower()] + corpus

    try:
        vec = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        mat  = vec.fit_transform(all_texts)
        sims = cosine_similarity(mat[0:1], mat[1:]).flatten()
        for rank, orig_i in enumerate(scored_idx):
            scores[orig_i] = round(float(sims[rank]), 4)
    except Exception as exc:
        logger.warning('TF-IDF failed: %s — using neutral scores', exc)

    return scores


# ── Main entry point ───────────────────────────────────────────────────────────

def compute_matches(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Compute and return a ranked list of workers by composite score.

    Expected payload (Django → ML Service):
    {
        "job_request": {
            "job_type_name":  str,          # name of the selected job type tile
            "description":    str,          # optional resident notes
            "budget_min":     float | null,
            "budget_max":     float | null,
            "location_lat":   float | null,
            "location_lng":   float | null
        },
        "candidates": [
            {
                "worker_id":        str,
                "declared_rate":    float,
                "avg_rating":       float,
                "years_experience": int,
                "address_lat":      float | null,
                "address_lng":      float | null,
                "bio":              str | null   # optional
            },
            ...
        ]
    }
    """
    job   = payload['job_request']
    cands = payload['candidates']

    if not cands:
        return []

    job_lat       = job.get('location_lat')
    job_lng       = job.get('location_lng')
    budget_min    = job.get('budget_min')
    budget_max    = job.get('budget_max')
    job_type_name = job.get('job_type_name', '')
    notes         = job.get('description', '')

    query = _build_query(job_type_name, notes)
    bios  = [str(c.get('bio') or '') for c in cands]
    t_scores = _text_scores(query, bios)

    results = []
    for i, c in enumerate(cands):
        t  = t_scores[i]
        pr = _proximity_score(job_lat, job_lng, c.get('address_lat'), c.get('address_lng'))
        p  = _price_score(budget_min, budget_max, float(c.get('declared_rate', 0)))
        r  = _rating_score(float(c.get('avg_rating', 0)))
        e  = _experience_score(int(c.get('years_experience', 0)))

        composite = round(
            WEIGHT_TEXT       * t  +
            WEIGHT_PROXIMITY  * pr +
            WEIGHT_PRICE      * p  +
            WEIGHT_RATING     * r  +
            WEIGHT_EXPERIENCE * e,
            4
        )

        results.append({
            'worker_id': c['worker_id'],
            'score':     composite,
            'score_breakdown': {
                'text_score':        t,
                'proximity_score':   pr,
                'price_score':       p,
                'rating_score':      r,
                'experience_score':  e,
            },
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results