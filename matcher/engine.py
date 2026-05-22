# Skill-Link-Cdo-ML / matcher/engine.py
#
# Core Matching Engine — Skill-Link CDO ML Service
#
# MODEL: Hybrid Content-Based Recommender Model
# -----------------------------------------------
#
# This is a Weighted Linear Combination Model — the standard baseline
# for hybrid recommender systems (Ricci et al., 2022).

from __future__ import annotations
import math
import re
import logging
 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
from config import (
    WEIGHT_TEXT,
    WEIGHT_PROXIMITY,
    WEIGHT_PRICE,
    WEIGHT_RATING,
    MAX_PROXIMITY_KM,
    FALLBACK_TEXT_SCORE,
)
from matcher.schema import JobRequestPayload, CandidateWorker, RankedWorker
 
logger = logging.getLogger("skilllink-ml.engine")
 
_EARTH_RADIUS_KM = 6371.0
 
 
# Text normalization
 
def normalize_text(text: str) -> str:
    """
    Lowercase, remove punctuation, collapse whitespace.
    """
    if not text or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
 
 
# Signal 1: TF-IDF + Cosine Similarity
 
def compute_text_scores(query: str, bios: list[str]) -> list[float]:
    """
    Vectorizes the job description (query) and all worker bios using TF-IDF,
    then computes Cosine Similarity between the query and each bio vector.
 
    Returns a list of float scores in [0.0, 1.0], one per candidate.
    """
    has_content = any(b.strip() for b in bios)
    if not has_content:
        logger.warning(
            "All candidate bios are empty. Applying fallback text score %.2f.",
            FALLBACK_TEXT_SCORE,
        )
        return [FALLBACK_TEXT_SCORE] * len(bios)
 
    corpus = [query] + bios
 
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
 
    query_vector = tfidf_matrix[0]
    bio_vectors  = tfidf_matrix[1:]
 
    scores = cosine_similarity(query_vector, bio_vectors).flatten()
    return scores.tolist()
 
 
# Signal 2: Haversine Proximity
 
def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Computes the great-circle distance in km between two coordinates.
    Uses the Haversine formula which accounts for Earth's curvature.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
 
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))
 
 
def proximity_score(distance_km: float) -> float:
    """
    Converts distance to a [0, 1]
    """
    return max(0.0, 1.0 - distance_km / MAX_PROXIMITY_KM)
 
 
# Signal 3: Price Compatibility
 
def price_score(
    declared_rate: float,
    budget_min: float | None,
    budget_max: float | None,
) -> float:
    """
    Scores how well the worker's declared rate fits the resident's
    selected budget range.
 
    - No budget specified → 0.5 (neutral; budget is not a factor)
    - Rate within [budget_min, budget_max] → 1.0 (perfect fit)
    - Rate below budget_min → gentle penalty (may still be acceptable)
    - Rate above budget_max → stronger penalty (resident likely cannot afford)
    """
    if budget_min is None and budget_max is None:
        return 0.5
 
    lo = budget_min if budget_min is not None else 0.0
    hi = budget_max if budget_max is not None else float("inf")
 
    if lo <= declared_rate <= hi:
        return 1.0
 
    if declared_rate < lo:
        penalty = min((lo - declared_rate) / max(lo, 1.0), 1.0)
        return max(0.0, 0.7 - 0.7 * penalty)
 
    # Above budget_max
    penalty = min((declared_rate - hi) / max(hi, 1.0), 1.0)
    return max(0.0, 1.0 - penalty)
 
 
# Signal 4: Average Rating
 
def rating_score(avg_rating: float) -> float:
    """
    Normalizes WorkerProfile.avg_rating (0.00–5.00) to [0.0, 1.0].
 
    Workers with avg_rating == 0.0 have no reviews yet (new workers).
    They receive 0.5 (neutral) instead of 0.0 to avoid unfairly
    penalizing newly registered workers.
    """
    if avg_rating == 0.0:
        return 0.5
    return avg_rating / 5.0
 
 
# Main entry point
 
def run_matching(
    job: JobRequestPayload,
    candidates: list[CandidateWorker],
) -> list[RankedWorker]:
    """
    Runs the full matching pipeline and returns workers ranked by
    composite score descending.
 
    Pipeline:
      1. Normalize query text and all worker bios.
      2. Compute TF-IDF cosine similarity scores.
      3. For each candidate, compute proximity, price, rating scores.
      4. Combine with Weighted Linear Combination.
      5. Sort descending and return.
    """
    query = normalize_text(job.job_description)
    bios  = [normalize_text(c.bio or "") for c in candidates]
 
    text_scores = compute_text_scores(query, bios)
 
    results: list[RankedWorker] = []
 
    for i, candidate in enumerate(candidates):
 
        # Proximity — 0.0 if worker has no stored coordinates
        if candidate.address_lat is not None and candidate.address_lng is not None:
            dist_km = haversine_km(
                job.location_lat, job.location_lng,
                candidate.address_lat, candidate.address_lng,
            )
            prox = proximity_score(dist_km)
        else:
            prox = 0.0
            logger.debug(
                "Worker %s has no coordinates — proximity score set to 0.0.",
                candidate.worker_id,
            )
 
        t_score = text_scores[i]
        p_score = price_score(candidate.declared_rate, job.budget_min, job.budget_max)
        r_score = rating_score(candidate.avg_rating)
 
        composite = (
            WEIGHT_TEXT      * t_score
            + WEIGHT_PROXIMITY * prox
            + WEIGHT_PRICE     * p_score
            + WEIGHT_RATING    * r_score
        )
        composite = max(0.0, min(1.0, composite))
 
        results.append(
            RankedWorker(
                worker_id=candidate.worker_id,
                score=round(composite, 6),
                score_breakdown={
                    "text_score":      round(t_score, 6),
                    "proximity_score": round(prox, 6),
                    "price_score":     round(p_score, 6),
                    "rating_score":    round(r_score, 6),
                },
            )
        )
 
    results.sort(key=lambda r: r.score, reverse=True)
    return results