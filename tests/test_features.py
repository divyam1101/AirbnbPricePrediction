import numpy as np
import pandas as pd
import pytest

from airbnb_price import features as F
from airbnb_price.evaluate import regression_metrics


def cleaned_rows(city: str, prices: list[float]) -> pd.DataFrame:
    n = len(prices)
    data = {
        "city": [city] * n,
        "price": prices,
        "security_deposit": [100] * n,
        "cleaning_fee": [10] * n,
        "extra_people": [0] * n,
        "host_response_time": ["within an hour"] * (n - 1) + ["unknown"],
        "host_response_rate": ["100"] * (n - 1) + ["unknown"],
        "property_type": ["Apartment"] * n,
        "room_type": ["Entire home/apt"] * n,
        "bed_type": ["Real Bed"] * n,
        "cancellation_policy": ["luxury_super_strict_95"] + ["flexible"] * (n - 1),
        "review_scores_rating": ["95-100"] * (n - 1) + [np.nan],
        **{col: ["9"] * n for col in F.TEN_POINT_REVIEWS},
        **{col: [1.0] * n for col in F.NUMERIC_COLUMNS if col not in F.MONEY_COLUMNS},
        **{flag: [1.0] + [np.nan] * (n - 1) for flag in F.AMENITY_FLAGS},
    }
    return pd.DataFrame(data)


def test_london_prices_are_converted_to_euros():
    out = F.build_features(cleaned_rows("london", [100, 200]))
    assert out["price_eur"].tolist() == pytest.approx([112.0, 224.0])
    assert out["security_deposit"].iloc[0] == pytest.approx(112.0)
    assert out[F.TARGET].iloc[0] == pytest.approx(np.log(112.0))


def test_out_of_scope_prices_are_removed():
    out = F.build_features(cleaned_rows("paris", [0, 5, 50, 5000]))
    assert out["price_eur"].tolist() == [50.0]


def test_types_and_encodings():
    out = F.build_features(cleaned_rows("berlin", [50, 60, 70]))
    assert out["air_conditioning"].tolist() == [1, 0, 0]
    assert str(out["air_conditioning"].dtype) == "int8"
    assert out["review_scores_cleanliness"].tolist() == [2.0, 2.0, 2.0]
    assert np.isnan(out["review_scores_rating"].iloc[-1])
    assert out["has_reviews"].tolist() == [1, 1, 0]
    assert np.isnan(out["host_response_rate"].iloc[-1])
    assert out["cancellation_policy"].iloc[0] == "strict_14_with_grace_period"
    assert isinstance(out["room_type"].dtype, pd.CategoricalDtype)
    assert set(F.feature_columns()) <= set(out.columns)


def test_unknown_city_currency_raises():
    with pytest.raises(ValueError):
        F.build_features(cleaned_rows("rome", [50, 60]))


def test_align_categories_shares_levels():
    a = pd.DataFrame({c: pd.Categorical(["x"]) for c in F.NOMINAL_COLUMNS})
    b = pd.DataFrame({c: pd.Categorical(["y"]) for c in F.NOMINAL_COLUMNS})
    a2, b2 = F.align_categories([a, b])
    assert list(a2["room_type"].cat.categories) == ["x", "y"] == list(b2["room_type"].cat.categories)


def test_regression_metrics():
    m = regression_metrics(np.log([100, 200]), np.log([110, 180]))
    assert m["mae_eur"] == pytest.approx(15.0)
    assert m["median_ape"] == pytest.approx(0.1)
    assert m["n"] == 2
