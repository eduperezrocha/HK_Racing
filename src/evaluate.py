# src/evaluate.py

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss, brier_score_loss


def evaluate_probabilities(
    df: pd.DataFrame,
    target_col: str = "won",
    prob_col: str = "model_prob",
) -> dict:
    """
    Evaluate probability quality using log loss and Brier score.
    Lower is better for both.
    """

    clean = df[[target_col, prob_col]].dropna().copy()

    return {
        "log_loss": log_loss(clean[target_col], clean[prob_col]),
        "brier_score": brier_score_loss(clean[target_col], clean[prob_col]),
    }


def compare_model_to_market(
    df: pd.DataFrame,
    target_col: str = "won",
    model_prob_col: str = "model_prob",
    market_prob_col: str = "market_prob",
) -> pd.DataFrame:
    """
    Compare your model probabilities against market-implied probabilities.

    Main goal:
        model log loss < market log loss
    """

    clean = df[[target_col, model_prob_col, market_prob_col]].dropna().copy()

    model_log_loss = log_loss(clean[target_col], clean[model_prob_col])
    market_log_loss = log_loss(clean[target_col], clean[market_prob_col])

    model_brier = brier_score_loss(clean[target_col], clean[model_prob_col])
    market_brier = brier_score_loss(clean[target_col], clean[market_prob_col])

    results = pd.DataFrame([
        {
            "source": "model",
            "log_loss": model_log_loss,
            "brier_score": model_brier,
        },
        {
            "source": "market",
            "log_loss": market_log_loss,
            "brier_score": market_brier,
        },
    ])

    return results


def check_race_probability_sums(
    df: pd.DataFrame,
    race_col: str = "race_id",
    prob_col: str = "model_prob",
) -> pd.DataFrame:
    """
    Check whether model probabilities sum to 1 inside each race.
    """

    sums = (
        df.groupby(race_col)[prob_col]
        .sum()
        .reset_index(name="prob_sum")
    )

    return sums.describe()


def calibration_table(
    df: pd.DataFrame,
    target_col: str = "won",
    prob_col: str = "model_prob",
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Create calibration table.

    avg_predicted_prob:
        average probability predicted by the model

    actual_win_rate:
        actual percentage of horses that won in that bucket

    count:
        number of horses in that bucket
    """

    clean = df[[target_col, prob_col]].dropna().copy()

    clean["prob_bucket"] = pd.qcut(
        clean[prob_col],
        q=n_bins,
        duplicates="drop",
    )

    table = (
        clean.groupby("prob_bucket", observed=True)
        .agg(
            avg_predicted_prob=(prob_col, "mean"),
            actual_win_rate=(target_col, "mean"),
            count=(target_col, "size"),
        )
        .reset_index()
    )

    table["calibration_error"] = (
        table["actual_win_rate"] - table["avg_predicted_prob"]
    )

    return table


def plot_calibration_curve(
    calibration: pd.DataFrame,
    title: str = "Calibration Curve",
) -> None:
    """
    Plot calibration curve.

    Perfect calibration means the points are close to the diagonal line.
    """

    plt.figure(figsize=(6, 6))

    plt.plot(
        calibration["avg_predicted_prob"],
        calibration["actual_win_rate"],
        marker="o",
    )

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("Average predicted probability")
    plt.ylabel("Actual win rate")
    plt.title(title)
    plt.show()


def top_predictions(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    prob_col: str = "model_prob",
    n: int = 20,
) -> pd.DataFrame:
    """
    Show highest model probability horses.
    """

    if columns is None:
        columns = [
            "race_id",
            "horse_id",
            "won",
            "win_odds",
            "model_prob",
            "market_prob",
            "expected_profit",
        ]

    available_columns = [col for col in columns if col in df.columns]

    return (
        df[available_columns]
        .sort_values(prob_col, ascending=False)
        .head(n)
    )


def top_value_bets(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    value_col: str = "expected_profit",
    n: int = 20,
) -> pd.DataFrame:
    """
    Show horses with the highest expected profit.
    """

    if columns is None:
        columns = [
            "race_id",
            "horse_id",
            "won",
            "win_odds",
            "model_prob",
            "market_prob",
            "edge",
            "expected_profit",
        ]

    available_columns = [col for col in columns if col in df.columns]

    return (
        df[available_columns]
        .sort_values(value_col, ascending=False)
        .head(n)
    )