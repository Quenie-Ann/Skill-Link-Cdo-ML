# Skill-Link-Cdo-ML / config.py
#
# Scoring Weight Configuration — Skill-Link CDO ML Service
# RATIONALE FOR DEFAULT WEIGHTS:
# ──────────────────────────────
# Text relevance (0.35):
#   Highest weight because the job type description + worker bio are the
#   most semantically meaningful matching signals. The Admin-defined job
#   type description removes free-text noise, making TF-IDF highly reliable.
#
# Proximity (0.30):
#   Second-highest. Barangay-scoped labor matching is inherently local.
#   Residents strongly prefer workers nearby. Consistent with the design
#   objective of reducing hiring friction within the community.
#
# Rating (0.20):
#   Third. Average rating reflects demonstrated reliability across prior
#   jobs. Weighted moderately — new workers with 0 ratings receive a
#   neutral score (0.5) to avoid unfair penalization (see engine.py).
#
# Price (0.15):
#   Lowest. Budget range is marked as a "soft signal".
#   Residents may accept rates slightly outside their stated range.
#   Overly penalizing price deviation would surface cheaper but unqualified
#   workers over better-matched, slightly pricier ones.

WEIGHT_TEXT:      float = 0.35
WEIGHT_PROXIMITY: float = 0.30
WEIGHT_PRICE:     float = 0.15
WEIGHT_RATING:    float = 0.20
 
assert abs((WEIGHT_TEXT + WEIGHT_PROXIMITY + WEIGHT_PRICE + WEIGHT_RATING) - 1.0) < 1e-9, (
    f"Weights must sum to 1.0. Current sum: "
    f"{WEIGHT_TEXT + WEIGHT_PROXIMITY + WEIGHT_PRICE + WEIGHT_RATING}"
)
 
# Workers beyond this distance score 0.0 for proximity.
# 15 km covers a single CDO barangay deployment generously.
MAX_PROXIMITY_KM: float = 15.0
 
# Applied when ALL candidate bios are empty strings.
# Slightly below neutral so workers without bios rank below those with bios.
FALLBACK_TEXT_SCORE: float = 0.4