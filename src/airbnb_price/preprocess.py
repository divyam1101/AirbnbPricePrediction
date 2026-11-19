"""Clean a raw Inside Airbnb ``listings.csv.gz`` snapshot into the per-city format in ``data/``.

This replaces the 2021 preprocessing notebook, which had to be re-run by hand for each city
(editing the S3 path and output name each time) and depended on SageMaker. Improvements:

* One function, parameterised by snapshot date, that can be unit-tested and run from the CLI.
* Amenities are parsed into exact names before matching. The old substring regexes produced false
  positives: ``"pet"`` matched "Pets live on this property" (the host has pets, which does not mean
  guests may bring them), ``"TV"`` matched every TV variant, and ``"Cat(s)"`` was read as a regex
  group.
* Money columns are parsed robustly (any currency symbol, thousands separators, missing values).

Usage::

    python -m airbnb_price.preprocess raw/listings_paris.csv.gz data/paris_cleaned_v2.csv.gz --snapshot 2019-09-15

The ``*_cleaned_v2.csv.gz`` files shipped in ``data/`` were produced by the 2021 notebook. Re-running
this module on the same raw snapshots gives the same columns, with the corrected amenity flags.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

AMENITY_GROUPS: dict[str, set[str]] = {
    "check_in_24h": {"24-hour check-in"},
    "air_conditioning": {"air conditioning", "central air conditioning"},
    "high_end_electronics": {"amazon echo", "apple tv", "game console", "netflix", "projector and screen", "smart tv"},
    "bbq": {"bbq grill", "fire pit", "propane barbeque"},
    "balcony": {"balcony", "patio or balcony"},
    "nature_and_views": {"beach view", "beachfront", "lake access", "mountain view", "ski-in/ski-out", "waterfront"},
    "bed_linen": {"bed linens"},
    "breakfast": {"breakfast"},
    "tv": {"tv", "cable tv", "smart tv"},
    "coffee_machine": {"coffee maker", "espresso machine"},
    "cooking_basics": {"cooking basics"},
    "white_goods": {"dishwasher", "dryer", "washer", "washer / dryer"},
    "elevator": {"elevator"},
    "gym": {"gym", "exercise equipment", "shared gym", "private gym"},
    "child_friendly": {"family/kid friendly", "children’s books and toys", "children's books and toys",
                       "children’s dinnerware", "children's dinnerware"},
    "parking": {"free parking on premises", "free street parking", "paid parking off premises",
                "paid parking on premises"},
    "outdoor_space": {"garden or backyard", "outdoor seating", "sun loungers", "terrace"},
    "host_greeting": {"host greets you"},
    "hot_tub_sauna_or_pool": {"hot tub", "jetted tub", "private hot tub", "sauna", "pool", "private pool"},
    "internet": {"internet", "wifi", "pocket wifi"},
    "long_term_stays": {"long term stays allowed"},
    "pets_allowed": {"pets allowed"},
    "private_entrance": {"private entrance"},
    "secure": {"safe", "security system", "lock on bedroom door"},
    "self_check_in": {"self check-in"},
    "smoking_allowed": {"smoking allowed"},
    "accessible": {"step-free access", "wheelchair accessible", "accessible-height bed", "accessible-height toilet"},
    "event_suitable": {"suitable for events"},
}

PROPERTY_TYPE_MAP = {
    "Townhouse": "House", "Bungalow": "House", "Cottage": "House", "Villa": "House",
    "Tiny house": "House", "Earth house": "House", "Chalet": "House",
    "Serviced apartment": "Apartment", "Aparthotel": "Apartment", "Condominium": "Apartment",
    "Loft": "Apartment", "Boutique hotel": "Hotel",
}

KEEP_COLUMNS = [
    "id", "experiences_offered", "host_response_time", "host_response_rate", "host_is_superhost",
    "host_listings_count", "host_identity_verified", "neighbourhood_cleansed", "city", "country",
    "property_type", "room_type", "accommodates", "bathrooms", "bedrooms", "beds", "bed_type",
    "amenities", "price", "security_deposit", "cleaning_fee", "guests_included", "extra_people",
    "minimum_nights", "maximum_nights", "availability_30", "availability_60", "availability_90",
    "availability_365", "number_of_reviews", "review_scores_rating", "review_scores_accuracy",
    "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication",
    "review_scores_location", "review_scores_value", "instant_bookable", "cancellation_policy",
    "host_since",
]

BOOLEAN_COLUMNS = ["host_is_superhost", "host_identity_verified", "instant_bookable"]
TEN_POINT_REVIEWS = [
    "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin",
    "review_scores_communication", "review_scores_location", "review_scores_value",
]


def parse_money(series: pd.Series) -> pd.Series:
    """'$1,234.00' / '£85.00' / NaN -> 1234.0 / 85.0 / NaN."""
    cleaned = series.astype("string").str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned.replace("", pd.NA), errors="coerce").astype("float64")


def parse_amenities(value) -> set[str]:
    """'{TV,"Cable TV",Wifi}' -> {'tv', 'cable tv', 'wifi'} (lower-cased exact names)."""
    if not isinstance(value, str):
        return set()
    items = re.findall(r'"([^"]*)"|([^,{}"]+)', value)
    return {(quoted or bare).strip().lower() for quoted, bare in items if (quoted or bare).strip()}


def amenity_flags(amenities: pd.Series) -> pd.DataFrame:
    parsed = amenities.map(parse_amenities)
    return pd.DataFrame(
        {flag: parsed.map(lambda s, names=names: int(bool(s & names))) for flag, names in AMENITY_GROUPS.items()},
        index=amenities.index,
    ).astype("int8")


def _bin(series: pd.Series, bins, labels, missing: str) -> pd.Series:
    binned = pd.cut(pd.to_numeric(series, errors="coerce"), bins=bins, labels=labels, include_lowest=True)
    return binned.astype("object").where(binned.notna(), missing)


def clean_listings(raw: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    """Turn one raw Inside Airbnb listings table into the cleaned modelling format."""
    df = raw[[c for c in KEEP_COLUMNS if c in raw.columns]].copy()
    df = df.rename(columns={"neighbourhood_cleansed": "borough"})

    for col in BOOLEAN_COLUMNS:
        df[col] = df[col].map({"t": 1, "f": 0}).astype("float64")

    host_since = pd.to_datetime(df.pop("host_since"), errors="coerce")
    df = df[host_since.notna()].copy()
    df["host_days_active"] = (pd.Timestamp(snapshot_date) - host_since[host_since.notna()]).dt.days.astype("float64")

    df["host_response_time"] = df["host_response_time"].fillna("unknown")
    rate = pd.to_numeric(df["host_response_rate"].astype("string").str.rstrip("%"), errors="coerce")
    df["host_response_rate"] = _bin(rate, [0, 25, 50, 75, 99, 100],
                                    ["0-24", "25-49", "50-74", "75-99", "100"], "unknown")

    df["property_type"] = df["property_type"].replace(PROPERTY_TYPE_MAP)
    df.loc[~df["property_type"].isin(["House", "Apartment", "Hotel"]), "property_type"] = "Other"

    for col in ["bathrooms", "bedrooms", "beds"]:
        df[col] = df[col].fillna(df[col].median())

    df["price"] = parse_money(df["price"])
    for col in ["security_deposit", "cleaning_fee", "extra_people"]:
        df[col] = parse_money(df[col]).fillna(0.0)

    for col in TEN_POINT_REVIEWS:
        df[col] = _bin(df[col], [0, 7, 8, 9, 10], ["0-7", "8", "9", "10"], "no reviews")
    df["review_scores_rating"] = _bin(df["review_scores_rating"], [0, 50, 80, 95, 100],
                                      ["0-49", "50-79", "80-94", "95-100"], "no reviews")

    df = pd.concat([df, amenity_flags(df["amenities"])], axis=1)
    df = df[df["price"].notna()]
    return df.reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Clean a raw Inside Airbnb listings file.")
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--snapshot", required=True, help="date the snapshot was scraped, YYYY-MM-DD")
    args = parser.parse_args(argv)
    cleaned = clean_listings(pd.read_csv(args.raw, low_memory=False), args.snapshot)
    cleaned.to_csv(args.output, index=False)
    print(f"Wrote {len(cleaned):,} listings to {args.output}")


if __name__ == "__main__":
    main()


__all__ = ["clean_listings", "parse_money", "parse_amenities", "amenity_flags", "np"]
