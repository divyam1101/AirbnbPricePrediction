"""Load the cleaned per-city listings and turn them into a model-ready feature table.

Design choices (and fixes relative to the 2021 notebook):

* **One currency.** London prices are in GBP while the other cities are in EUR. The original
  model compared them directly; here every money column is converted to EUR.
* **Missing amenity flags mean "absent".** The cleaned files store amenities as 1/NaN. The
  original notebook filled NaN with the string ``"unkown"``, which silently turned 30 numeric
  columns into strings that were then label-encoded. They are now 0/1 integers.
* **Ordered bins stay ordered.** Review-score and response-rate bins are encoded as ordinal
  integers (missing = NaN), not arbitrary label codes.
* **No city identifiers as features.** City GDP was a constant per city, so with three training
  cities it acted as a city ID, and tree models cannot extrapolate it to a fourth city.
  Generalisation to new cities is measured instead with leave-one-city-out validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CITIES = ("amsterdam", "berlin", "paris", "london")
CURRENCY = {"amsterdam": "EUR", "berlin": "EUR", "paris": "EUR", "london": "GBP"}
# Inside Airbnb snapshots used here are from September 2019; average GBP->EUR rate that month.
FX_TO_EUR = {"EUR": 1.0, "GBP": 1.12}

MONEY_COLUMNS = ["price", "security_deposit", "cleaning_fee", "extra_people"]

# Business scope of the model: typical short-stay listings. Applied identically to every city
# *before* any split, so it is a definition of the population, not a peek at the test set.
PRICE_RANGE_EUR = (10.0, 1000.0)

AMENITY_FLAGS = [
    "check_in_24h", "air_conditioning", "high_end_electronics", "bbq", "balcony",
    "nature_and_views", "bed_linen", "breakfast", "tv", "coffee_machine", "cooking_basics",
    "white_goods", "elevator", "gym", "child_friendly", "parking", "outdoor_space",
    "host_greeting", "hot_tub_sauna_or_pool", "internet", "long_term_stays", "pets_allowed",
    "private_entrance", "secure", "self_check_in", "smoking_allowed", "accessible", "event_suitable",
]

TEN_POINT_REVIEWS = [
    "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin",
    "review_scores_communication", "review_scores_location", "review_scores_value",
]
ORDINAL_MAPS = {
    **{col: {"0-7": 0, "8": 1, "9": 2, "10": 3} for col in TEN_POINT_REVIEWS},
    "review_scores_rating": {"0-49": 0, "50-79": 1, "80-94": 2, "95-100": 3},
    "host_response_rate": {"0-24": 0, "25-49": 1, "50-74": 2, "75-99": 3, "100": 4},
    "host_response_time": {
        "within an hour": 0, "within a few hours": 1, "within a day": 2, "a few days or more": 3,
    },
}

NOMINAL_COLUMNS = ["property_type", "room_type", "bed_type", "cancellation_policy"]
CANCELLATION_ALIASES = {
    "super_strict_30": "strict_14_with_grace_period",
    "super_strict_60": "strict_14_with_grace_period",
    "strict": "strict_14_with_grace_period",
    "luxury_super_strict_95": "strict_14_with_grace_period",
    "luxury_moderate": "moderate",
}

NUMERIC_COLUMNS = [
    "host_is_superhost", "host_listings_count", "host_identity_verified", "accommodates",
    "bathrooms", "bedrooms", "beds", "security_deposit", "cleaning_fee", "guests_included",
    "extra_people", "minimum_nights", "maximum_nights", "availability_30", "availability_60",
    "availability_90", "availability_365", "number_of_reviews", "instant_bookable",
    "host_days_active",
]

DROP_COLUMNS = ["id", "amenities", "borough", "city", "country", "experiences_offered"]
TARGET = "log_price_eur"


def load_city(path: str | Path, city: str) -> pd.DataFrame:
    """Read one cleaned city file and add a ``city`` label."""
    df = pd.read_csv(path, low_memory=False)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df.insert(0, "city", city)
    return df


def load_all(data_dir: str | Path, cities: tuple[str, ...] = CITIES) -> pd.DataFrame:
    data_dir = Path(data_dir)
    frames = [load_city(data_dir / f"{city}_cleaned_v2.csv.gz", city) for city in cities]
    return pd.concat(frames, ignore_index=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy with EUR money columns, typed features and the log-price target.

    Rows outside :data:`PRICE_RANGE_EUR` are removed.
    """
    out = df.copy()

    fx = out["city"].map(CURRENCY).map(FX_TO_EUR)
    if fx.isna().any():
        unknown = sorted(out.loc[fx.isna(), "city"].unique())
        raise ValueError(f"no currency configured for cities: {unknown}")
    for col in MONEY_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce") * fx
    out = out.rename(columns={"price": "price_eur"})

    lo, hi = PRICE_RANGE_EUR
    out = out[out["price_eur"].between(lo, hi)].copy()
    out[TARGET] = np.log(out["price_eur"])

    for col in AMENITY_FLAGS:
        out[col] = out[col].fillna(0).astype("int8")

    for col, mapping in ORDINAL_MAPS.items():
        out[col] = out[col].map(mapping).astype("float64")  # unmapped ("unknown", NaN) -> NaN
    out["has_reviews"] = out["review_scores_rating"].notna().astype("int8")

    out["cancellation_policy"] = out["cancellation_policy"].replace(CANCELLATION_ALIASES)
    for col in NOMINAL_COLUMNS:
        out[col] = out[col].astype("category")

    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    return out.reset_index(drop=True)


def feature_columns() -> list[str]:
    return NUMERIC_COLUMNS + AMENITY_FLAGS + list(ORDINAL_MAPS) + ["has_reviews"] + NOMINAL_COLUMNS


def align_categories(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Give every nominal column the same category set across frames (train/test/new city)."""
    result = [f.copy() for f in frames]
    for col in NOMINAL_COLUMNS:
        levels = sorted(set().union(*(set(f[col].dropna().astype(str)) for f in frames)))
        for f in result:
            f[col] = pd.Categorical(f[col].astype("object"), categories=levels)
    return result
