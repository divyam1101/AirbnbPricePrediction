# Airbnb Price Prediction

Predicting nightly Airbnb listing prices across four European cities, and testing whether a model trained on some cities **generalises to a city it has never seen**.

## Problem

Hosts and platforms need a fair price estimate for a listing. Pricing depends on location, room type, amenities, host behaviour and reviews, and these relationships differ city by city. This project builds one model on **Amsterdam, Berlin and Paris** (~108k listings, 64 features) and evaluates it on held-out data **and** on **London** (~83k listings) as an unseen city.

## Approach

1. **Data preprocessing** (`Airbnb-price-prediction-Multiple cities-DataPreprocessing.ipynb`)
   - Cleaned raw Inside Airbnb listings for each city
   - Engineered amenity flags (e.g. air conditioning, elevator, balcony, coffee machine) and binned review scores
   - Added city-level context such as city GDP
2. **Modelling** (`Airbnb-price-prediction-Modeling-Final.ipynb`)
   - Log-transformed price, removed outliers
   - Dropped highly correlated and low-variance features
   - Label-encoded categorical variables
   - 80/20 train/validation split; London kept entirely out of training
   - Compared **TabNet** (deep learning for tabular data), **XGBoost**, **Random Forest** and **AdaBoost**

## Results (target = log price)

| Model | Validation R² | Validation MSE | London R² (unseen city) | London MSE |
|---|---|---|---|---|
| **XGBoost** | **0.749** | **0.109** | **0.630** | **0.208** |
| Random Forest | 0.720 | 0.122 | 0.582 | 0.235 |
| TabNet | 0.658 | 0.148 | 0.625 | 0.211 |
| AdaBoost | 0.496 | 0.218 | 0.409 | 0.333 |

**Takeaways**
- XGBoost gave the best accuracy both in-sample and on the unseen city.
- Random Forest overfits (train R² 0.96 vs validation 0.72) and transfers worst of the strong models.
- TabNet is weaker on validation but generalises to London almost as well as XGBoost, so it is the most stable across cities.
- Influential features include room type, cleanliness score, host responsiveness, city GDP, air conditioning and instant bookability.

## Data

Cleaned per-city datasets are included as `*_cleaned_v2.csv.gz` (Amsterdam, Berlin, Paris, London), derived from [Inside Airbnb](http://insideairbnb.com/).

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · PyTorch TabNet (`pytorch-tabnet`) · matplotlib · seaborn · Jupyter

## How to run

```bash
pip install pandas numpy scikit-learn xgboost pytorch-tabnet torch matplotlib seaborn jupyter
jupyter notebook
```

Open the preprocessing notebook first, then `Airbnb-price-prediction-Modeling-Final.ipynb`.
