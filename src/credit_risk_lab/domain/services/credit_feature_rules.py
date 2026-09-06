"""Versioned business mappings used by deterministic feature engineering."""

CREDIT_SCORE_BANDS = [300, 579, 669, 739, 799, 900]
CREDIT_SCORE_BAND_LABELS = ["Poor", "Fair", "Good", "Very Good", "Excellent"]

AGE_GROUP_BINS = [18, 25, 35, 50, 65, 110]
AGE_GROUP_LABELS = ["18-25", "26-35", "36-50", "51-65", "65+"]

CREDIT_HISTORY_BINS = [0, 3, 7, 15, 40]
CREDIT_HISTORY_LABELS = ["0-3", "4-7", "8-15", "15+"]

LOAN_INTENT_RISK = {
    "MEDICAL": 3,
    "PERSONAL": 2,
    "VENTURE": 2,
    "EDUCATION": 1,
    "HOMEIMPROVEMENT": 1,
    "DEBTCONSOLIDATION": 2,
}

HOME_OWNERSHIP_RISK = {"OWN": 2, "MORTGAGE": 1, "RENT": 0, "OTHER": 0}

EDUCATION_LEVELS = {
    "High School": 0,
    "Associate": 1,
    "Bachelor": 2,
    "Master": 3,
    "Doctorate": 4,
}
