"""Run the full experiment and write metrics and figures to ``reports/``.

Usage::

    python -m airbnb_price.train --data-dir data --out-dir reports

Two evaluation protocols:

1. **Holdout + unseen city** (comparable with the 2021 project): train on 80% of
   Amsterdam, Berlin and Paris, test on the other 20% and on all of London.
2. **Leave-one-city-out**: for each of the four cities, train on the other three and test on
   the held-out city. This is the honest measure of how well the model transfers to a new market.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from . import features as F  # noqa: E402
from .evaluate import regression_metrics  # noqa: E402
from .models import MODELS, RANDOM_STATE, fit  # noqa: E402

TRAIN_CITIES = ("amsterdam", "berlin", "paris")
UNSEEN_CITY = "london"


def holdout_experiment(data: pd.DataFrame, model_names: list[str]) -> tuple[pd.DataFrame, dict]:
    cols = F.feature_columns()
    pool = data[data["city"].isin(TRAIN_CITIES)]
    unseen = data[data["city"] == UNSEEN_CITY]
    train, test = train_test_split(pool, test_size=0.2, random_state=RANDOM_STATE, stratify=pool["city"])
    train, test, unseen = F.align_categories([train, test, unseen])

    rows, fitted = [], {}
    for name in model_names:
        model = fit(name, train[cols], train[F.TARGET])
        fitted[name] = model
        for split, frame in (("train", train), ("test (AMS/BER/PAR)", test), ("unseen city (London)", unseen)):
            rows.append({"model": model.name, "split": split,
                         **regression_metrics(frame[F.TARGET], model.predict(frame[cols]))})
    return pd.DataFrame(rows), {"fitted": fitted, "train": train, "test": test, "unseen": unseen}


def leave_one_city_out(data: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    cols = F.feature_columns()
    rows = []
    for held_out in F.CITIES:
        train, test = F.align_categories([data[data["city"] != held_out], data[data["city"] == held_out]])
        for name in model_names:
            model = fit(name, train[cols], train[F.TARGET])
            rows.append({"model": model.name, "held_out_city": held_out,
                         **regression_metrics(test[F.TARGET], model.predict(test[cols]))})
    return pd.DataFrame(rows)


def plot_results(holdout: pd.DataFrame, loco: pd.DataFrame, ctx: dict, best: str, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cols = F.feature_columns()

    # 1. model comparison
    pivot = holdout[holdout["split"] != "train"].pivot(index="model", columns="split", values="r2_log")
    ax = pivot.sort_values(pivot.columns[0]).plot.barh(figsize=(8, 4))
    ax.set_xlabel("R² (log price)")
    ax.set_title("Model comparison: held-out listings vs unseen city")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(fig_dir / "model_comparison.png", dpi=120)
    plt.close()

    # 2. leave-one-city-out
    pivot = loco.pivot(index="held_out_city", columns="model", values="r2_log")
    ax = pivot.plot.bar(figsize=(9, 4), rot=0)
    ax.set_ylabel("R² (log price)")
    ax.set_title("Leave-one-city-out: train on three cities, test on the fourth")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "leave_one_city_out.png", dpi=120)
    plt.close()

    # 3. predicted vs actual for the best model
    model = ctx["fitted"][best]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    panels = (("Held-out listings", ctx["test"]), ("London (unseen)", ctx["unseen"]))
    for ax, (title, frame) in zip(axes, panels, strict=True):
        sample = frame.sample(min(len(frame), 5000), random_state=RANDOM_STATE)
        ax.scatter(np.exp(sample[F.TARGET]), np.exp(model.predict(sample[cols])), s=3, alpha=0.3)
        ax.plot([10, 1000], [10, 1000], "k--", lw=1)
        ax.set(xscale="log", yscale="log", title=title, xlabel="Actual price (€)")
    axes[0].set_ylabel("Predicted price (€)")
    fig.suptitle(f"{model.name}: predicted vs actual nightly price")
    plt.tight_layout()
    plt.savefig(fig_dir / "predicted_vs_actual.png", dpi=120)
    plt.close()

    # 4. permutation importance on the held-out test set (model-agnostic, unlike split counts)
    sample = ctx["test"].sample(min(len(ctx["test"]), 5000), random_state=RANDOM_STATE)
    result = permutation_importance(
        model.estimator, sample[cols], sample[F.TARGET],
        n_repeats=5, random_state=RANDOM_STATE, scoring="r2",
    )
    importance = pd.Series(result.importances_mean, index=cols).sort_values().tail(15)
    ax = importance.plot.barh(figsize=(7, 5))
    ax.set_xlabel("Drop in R² when the feature is shuffled")
    ax.set_title(f"Top 15 features by permutation importance ({model.name})")
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_importance.png", dpi=120)
    plt.close()
    importance.sort_values(ascending=False).to_csv(out_dir / "feature_importance.csv", header=["r2_drop"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--out-dir", default="reports", type=Path)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = F.build_features(F.load_all(args.data_dir))
    print(f"Loaded {len(data):,} listings in scope "
          f"({F.PRICE_RANGE_EUR[0]:.0f}-{F.PRICE_RANGE_EUR[1]:.0f} EUR/night)")
    print(data["city"].value_counts().to_string(), "\n")

    holdout, ctx = holdout_experiment(data, args.models)
    print(holdout.round(3).to_string(index=False), "\n")
    loco = leave_one_city_out(data, args.models)
    print(loco.round(3).to_string(index=False), "\n")

    test_scores = holdout[holdout["split"] == "test (AMS/BER/PAR)"].set_index("model")["r2_log"]
    best_label = test_scores.idxmax()
    best = next(k for k, m in ctx["fitted"].items() if m.name == best_label)
    plot_results(holdout, loco, ctx, best, args.out_dir)

    summary = {
        "n_listings": int(len(data)),
        "price_range_eur": F.PRICE_RANGE_EUR,
        "fx_to_eur": F.FX_TO_EUR,
        "best_model": best_label,
        "holdout": holdout.to_dict(orient="records"),
        "leave_one_city_out": loco.to_dict(orient="records"),
        "leave_one_city_out_mean_r2": loco.groupby("model")["r2_log"].mean().round(4).to_dict(),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"Best model on held-out listings: {best_label}. Results written to {args.out_dir}/")


if __name__ == "__main__":
    main()
