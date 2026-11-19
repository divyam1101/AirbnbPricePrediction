import numpy as np
import pandas as pd
import pytest

from airbnb_price.preprocess import amenity_flags, clean_listings, parse_amenities, parse_money


def raw_listings() -> pd.DataFrame:
    base = {
        "id": [1, 2, 3],
        "experiences_offered": ["none"] * 3,
        "host_response_time": ["within an hour", None, "within a day"],
        "host_response_rate": ["100%", None, "60%"],
        "host_is_superhost": ["t", "f", "f"],
        "host_listings_count": [1, 3, 1],
        "host_identity_verified": ["t", "t", "f"],
        "neighbourhood_cleansed": ["Centrum", "Noord", "West"],
        "city": ["Amsterdam"] * 3,
        "country": ["Netherlands"] * 3,
        "property_type": ["Loft", "Castle", "Townhouse"],
        "room_type": ["Entire home/apt", "Private room", "Entire home/apt"],
        "accommodates": [2, 1, 4],
        "bathrooms": [1.0, np.nan, 2.0],
        "bedrooms": [1.0, 1.0, np.nan],
        "beds": [1.0, 1.0, 2.0],
        "bed_type": ["Real Bed"] * 3,
        "amenities": [
            '{TV,Wifi,"Pets live on this property","Cat(s)"}',
            '{"Smart TV","Pets allowed","Free street parking"}',
            "{}",
        ],
        "price": ["$1,250.00", "$85.00", "£40.00"],
        "security_deposit": ["$100.00", None, "$0.00"],
        "cleaning_fee": [None, "$20.00", "$15.00"],
        "guests_included": [1, 1, 2],
        "extra_people": ["$0.00", "$10.00", None],
        "minimum_nights": [2, 1, 3],
        "maximum_nights": [30, 1125, 60],
        "availability_30": [0, 10, 30],
        "availability_60": [0, 20, 60],
        "availability_90": [0, 30, 90],
        "availability_365": [0, 100, 365],
        "number_of_reviews": [10, 0, 5],
        "review_scores_rating": [96, np.nan, 70],
        "review_scores_accuracy": [10, np.nan, 7],
        "review_scores_cleanliness": [9, np.nan, 8],
        "review_scores_checkin": [10, np.nan, 9],
        "review_scores_communication": [10, np.nan, 9],
        "review_scores_location": [10, np.nan, 8],
        "review_scores_value": [9, np.nan, 7],
        "instant_bookable": ["f", "t", "f"],
        "cancellation_policy": ["moderate", "flexible", "strict"],
        "host_since": ["2019-09-05", "2018-01-01", None],
    }
    return pd.DataFrame(base)


def test_parse_money_handles_symbols_separators_and_missing():
    result = parse_money(pd.Series(["$1,234.50", "£85.00", None, "€0.00"]))
    assert result.iloc[0] == pytest.approx(1234.5)
    assert result.iloc[1] == pytest.approx(85.0)
    assert np.isnan(result.iloc[2])
    assert result.iloc[3] == 0.0


def test_parse_amenities_respects_quotes():
    assert parse_amenities('{TV,"Cable TV","Washer / Dryer"}') == {"tv", "cable tv", "washer / dryer"}
    assert parse_amenities(np.nan) == set()


def test_amenity_flags_use_exact_names():
    flags = amenity_flags(pd.Series(['{"Pets live on this property","Cat(s)"}', '{"Pets allowed"}']))
    # the host having pets does not mean guests may bring theirs (the 2021 regex said it did)
    assert flags["pets_allowed"].tolist() == [0, 1]


def test_clean_listings_end_to_end():
    cleaned = clean_listings(raw_listings(), snapshot_date="2019-09-15")

    assert len(cleaned) == 2, "rows without host_since are dropped"
    first = cleaned.iloc[0]
    assert first["price"] == pytest.approx(1250.0)
    assert first["host_days_active"] == 10
    assert first["host_response_rate"] == "100"
    assert first["property_type"] == "Apartment"  # Loft -> Apartment
    assert first["review_scores_rating"] == "95-100"
    assert first["review_scores_cleanliness"] == "9"
    assert first["cleaning_fee"] == 0.0  # missing fee means no fee
    assert first["tv"] == 1 and first["internet"] == 1 and first["pets_allowed"] == 0

    second = cleaned.iloc[1]
    assert second["property_type"] == "Other"  # Castle
    assert second["host_response_time"] == "unknown"
    assert second["review_scores_rating"] == "no reviews"
    assert second["bathrooms"] == pytest.approx(1.0)  # median of the rows that remain
    assert second["pets_allowed"] == 1 and second["parking"] == 1 and second["tv"] == 1
    assert "borough" in cleaned.columns and "neighbourhood_cleansed" not in cleaned.columns
