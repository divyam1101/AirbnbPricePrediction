"""Model zoo: a linear baseline and three gradient-boosted tree models.

Every model that uses early stopping takes its validation rows from the *training* data, never
from the test set. (The 2021 notebook early-stopped XGBoost on the test set, which made its
reported test scores optimistic.)

TabNet from the original project is not included: on tabular data like this, gradient-boosted
trees match or beat deep tabular models at a fraction of the cost (Grinsztajn et al., NeurIPS
2022; Shwartz-Ziv & Armon, 2022). The original run agreed: XGBoost beat TabNet.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
EARLY_STOPPING_FRACTION = 0.1


@dataclass
class FittedModel:
    name: str
    predict: Callable[[pd.DataFrame], np.ndarray]
    estimator: object


def _ridge(X: pd.DataFrame, y: pd.Series, seed: int) -> FittedModel:
    numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    categorical = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)
    pre = ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_exclude="category")),
        ("cat", categorical, make_column_selector(dtype_include="category")),
    ])
    model = make_pipeline(pre, RidgeCV(alphas=np.logspace(-3, 3, 13)))
    model.fit(X, y)
    return FittedModel("Ridge (baseline)", model.predict, model)


def _hist_gb(X: pd.DataFrame, y: pd.Series, seed: int) -> FittedModel:
    model = HistGradientBoostingRegressor(
        categorical_features="from_dtype", learning_rate=0.05, max_iter=2000,
        max_leaf_nodes=63, min_samples_leaf=30, l2_regularization=1.0,
        early_stopping=True, validation_fraction=EARLY_STOPPING_FRACTION, n_iter_no_change=50,
        random_state=seed,
    )
    model.fit(X, y)
    return FittedModel("HistGradientBoosting", model.predict, model)


def _lightgbm(X: pd.DataFrame, y: pd.Series, seed: int) -> FittedModel:
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=EARLY_STOPPING_FRACTION, random_state=seed)
    model = lgb.LGBMRegressor(
        n_estimators=3000, learning_rate=0.05, num_leaves=63, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=seed, verbose=-1,
    )
    model.fit(X_tr, y_tr, eval_X=X_val, eval_y=y_val, callbacks=[lgb.early_stopping(50, verbose=False)])
    return FittedModel("LightGBM", model.predict, model)


def _xgboost(X: pd.DataFrame, y: pd.Series, seed: int) -> FittedModel:
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=EARLY_STOPPING_FRACTION, random_state=seed)
    model = xgb.XGBRegressor(
        n_estimators=3000, learning_rate=0.05, max_depth=8, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
        enable_categorical=True, max_cat_to_onehot=1, early_stopping_rounds=50, random_state=seed, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return FittedModel("XGBoost", model.predict, model)


MODELS: dict[str, Callable[[pd.DataFrame, pd.Series, int], FittedModel]] = {
    "ridge": _ridge,
    "hist_gb": _hist_gb,
    "lightgbm": _lightgbm,
    "xgboost": _xgboost,
}


def fit(name: str, X: pd.DataFrame, y: pd.Series, seed: int = RANDOM_STATE) -> FittedModel:
    import time

    start = time.perf_counter()
    model = MODELS[name](X, y, seed)
    print(f"  fitted {model.name} on {len(X):,} rows in {time.perf_counter() - start:.0f}s", flush=True)
    return model
