# src/train.py

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from features import build_feature_dataset, get_feature_columns

from backtest import (
    add_market_probabilities,
    add_expected_value,
    run_threshold_backtests,
    get_strategy_bets,
    summarize_bets,
)

from evaluate import (
    compare_model_to_market,
    calibration_table,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def split_by_date(
    df: pd.DataFrame,
    train_size: float = 0.60,
    valid_size: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically into train, validation, and test.

    Train: oldest data
    Valid: middle data
    Test: newest data
    """

    unique_dates = df["date"].sort_values().unique()

    train_cutoff = unique_dates[int(len(unique_dates) * train_size)]
    valid_cutoff = unique_dates[int(len(unique_dates) * (train_size + valid_size))]

    train = df[df["date"] < train_cutoff].copy()

    valid = df[
        (df["date"] >= train_cutoff)
        & (df["date"] < valid_cutoff)
    ].copy()

    test = df[df["date"] >= valid_cutoff].copy()

    return train, valid, test


def build_model_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """
    Build preprocessing + XGBoost pipeline.
    """

    numeric_features = X_train.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = [
        col for col in X_train.columns
        if col not in numeric_features
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipe


def add_model_predictions(
    model: Pipeline,
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Predict raw probabilities and normalize them inside each race.
    """

    df = df.copy()

    X = df[features]

    df["raw_model_prob"] = model.predict_proba(X)[:, 1]

    race_sum = df.groupby("race_id")["raw_model_prob"].transform("sum")

    df["model_prob"] = np.where(
        race_sum > 0,
        df["raw_model_prob"] / race_sum,
        np.nan,
    )

    return df


def prepare_for_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add market probability, edge, expected return, and expected profit.
    """

    df = add_market_probabilities(df)
    df = add_expected_value(df)

    df["edge"] = df["model_prob"] - df["market_prob"]

    return df


def choose_best_threshold(
    threshold_results: pd.DataFrame,
    min_bets: int = 50,
    default_threshold: float = 0.20,
) -> float:
    """
    Choose best expected-profit threshold from validation results.

    Uses validation only, not test.
    """

    eligible = threshold_results[threshold_results["bets"] >= min_bets].copy()

    if eligible.empty:
        return default_threshold

    best_row = (
        eligible.sort_values(["roi", "profit"], ascending=False)
        .iloc[0]
    )

    return float(best_row["threshold"])


def main() -> None:
    print("Building feature dataset...")

    df = build_feature_dataset(DATA_DIR)
    features = get_feature_columns(df)

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("Features:", len(features))

    train, valid, test = split_by_date(df)

    print()
    print("Train:", train["date"].min(), "to", train["date"].max(), train.shape)
    print("Valid:", valid["date"].min(), "to", valid["date"].max(), valid.shape)
    print("Test:", test["date"].min(), "to", test["date"].max(), test.shape)

    X_train = train[features]
    y_train = train["won"]

    print()
    print("Training model...")

    model = build_model_pipeline(X_train)
    model.fit(X_train, y_train)

    print("Predicting validation and test sets...")

    valid_pred = add_model_predictions(model, valid, features)
    test_pred = add_model_predictions(model, test, features)

    valid_pred = prepare_for_backtest(valid_pred)
    test_pred = prepare_for_backtest(test_pred)

    print()
    print("Validation: model vs market")
    valid_comparison = compare_model_to_market(valid_pred)
    print(valid_comparison)

    print()
    print("Test: model vs market")
    test_comparison = compare_model_to_market(test_pred)
    print(test_comparison)

    print()
    print("Running validation threshold backtests...")

    thresholds = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 2.00]

    valid_threshold_results = run_threshold_backtests(
        valid_pred,
        thresholds=thresholds,
        max_odds=20,
    )

    print(valid_threshold_results)

    best_threshold = choose_best_threshold(valid_threshold_results)

    print()
    print("Chosen threshold from validation:", best_threshold)

    print()
    print("Testing chosen strategy on test set...")

    test_bets = get_strategy_bets(
        test_pred,
        threshold=best_threshold,
        max_odds=20,
    )

    test_summary = summarize_bets(test_bets)

    print("Test strategy summary:")
    for key, value in test_summary.items():
        print(f"{key}: {value}")

    print()
    print("Saving outputs...")

    joblib.dump(model, OUTPUT_DIR / "model.joblib")

    valid_pred.to_csv(OUTPUT_DIR / "valid_predictions.csv", index=False)
    test_pred.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    test_bets.to_csv(OUTPUT_DIR / "test_bets.csv", index=False)

    valid_threshold_results.to_csv(
        OUTPUT_DIR / "valid_threshold_results.csv",
        index=False,
    )

    valid_calibration = calibration_table(valid_pred)
    test_calibration = calibration_table(test_pred)

    valid_calibration.to_csv(
        OUTPUT_DIR / "valid_calibration.csv",
        index=False,
    )

    test_calibration.to_csv(
        OUTPUT_DIR / "test_calibration.csv",
        index=False,
    )

    print("Done.")
    print("Files saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()