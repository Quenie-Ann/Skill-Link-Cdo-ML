# Skill-Link-Cdo-ML / config.py
# skilllink-ml / config.py
#
# Scoring weights and tuning parameters.
# All values are configurable here without touching engine.py or main.py.
#
# Weight rules:
#   - All five weights must sum to 1.0
#   - Adjust per pilot feedback; re-deploy ML service only (Django untouched)

import os

# Scoring weights (must sum to 1.0) 
#
# Rationale for defaults:
#   Experience (0.30) — highest weight per scope decision: experience is the
#     primary trust signal for a barangay resident hiring without references.
#   Rating (0.25)     — strong signal but requires prior jobs; new workers
#     receive a neutral 0.5 so this does not unfairly exclude them early.
#   Proximity (0.20)  — barangay-scale system; distance matters but a highly
#     experienced worker 2km away should still beat a nearby inexperienced one.
#   Price (0.15)      — soft signal per SRS; budget is a preference not a gate.
#   Text (0.10)       — lowest weight because job_type_name is admin-defined
#     and already category-scoped; bio is optional and may be absent.

WEIGHT_EXPERIENCE : float = float(os.getenv('WEIGHT_EXPERIENCE', '0.30'))
WEIGHT_RATING     : float = float(os.getenv('WEIGHT_RATING',     '0.25'))
WEIGHT_PROXIMITY  : float = float(os.getenv('WEIGHT_PROXIMITY',  '0.20'))
WEIGHT_PRICE      : float = float(os.getenv('WEIGHT_PRICE',      '0.15'))
WEIGHT_TEXT       : float = float(os.getenv('WEIGHT_TEXT',       '0.10'))

# Proximity decay 
# Controls how fast the proximity score drops with distance.
# score = e^(-km / PROXIMITY_DECAY_KM)
#
# At PROXIMITY_DECAY_KM = 3.0:
#   0.0 km → 1.00    1.0 km → 0.72
#   2.0 km → 0.51    3.0 km → 0.37
#   5.0 km → 0.19   10.0 km → 0.04
#
# 3.0 km is appropriate for a single barangay pilot scope.
# Increase to 5.0 or 8.0 when the system expands city-wide.
PROXIMITY_DECAY_KM : float = float(os.getenv('PROXIMITY_DECAY_KM', '3.0'))

# Experience cap 
# Years at or beyond this value receive a full experience score of 1.0.
# Workers below this are scored on a square-root curve.
# Default 10 years: reflects the point where additional years yield
# negligible practical differentiation for barangay-level jobs.
EXPERIENCE_CAP_YEARS : int = int(os.getenv('EXPERIENCE_CAP_YEARS', '10'))