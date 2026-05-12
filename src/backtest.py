# src/backtest.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def add_market_probabilities(
    df: pd.DataFrame,
    race_col: str = "race_id",
    odds_col: str = "win_odds",
) -> pd.DataFrame:
    """
    Convert decimal odds into normalized market-implied probabilities.

    Example:
    win_odds = 5.0
    raw implied probability = 1 / 5.0 = 0.20

    Then probabilities are normalized inside each race so they sum to 1.
    """

    df = df.copy()

    df[odds_col] = pd.to_numeric(df[odds_col], errors="coerce")
    df = df[df[odds_col].notna()].copy()
    df = df[df[odds_col] > 0].copy()

    df["market_prob_raw"] = 1 / df[odds_col]

    df["market_prob"] = (
        df["market_prob_raw"] /
        df.groupby(race_col)["market_prob_raw"].transform("sum")
    )

    return df


def add_expected_value(
    df: pd.DataFrame,
    prob_col: str = "model_prob",
    odds_col: str = "win_odds",
) -> pd.DataFrame:
    """
    Calculate expected return and expected profit.

    expected_return = model probability * decimal odds
    expected_profit = expected_return - 1

    If expected_profit > 0, the model thinks the bet has positive value.
    """

    df = df.copy()

    df["expected_return"] = df[prob_col] * df[odds_col]
    df["expected_profit"] = df["expected_return"] - 1

    return df


def select_bets(
    df: pd.DataFrame,
    threshold: float = 0.20,
    min_odds: float | None = None,
    max_odds: float | None = None,
) -> pd.DataFrame:
    """
    Select bets where expected profit is above a threshold.

    Optional:
    - min_odds: ignore very low odds
    - max_odds: ignore extreme longshots
    """

    bets = df[df["expected_profit"] > threshold].copy()

    if min_odds is not None:
        bets = bets[bets["win_odds"] >= min_odds].copy()

    if max_odds is not None:
        bets = bets[bets["win_odds"] <= max_odds].copy()

    return bets


def calculate_bet_profits(
    bets: pd.DataFrame,
    target_col: str = "won",
    odds_col: str = "win_odds",
    stake: float = 1.0,
) -> pd.DataFrame:
    """
    Calculate profit for flat staking.

    If horse wins:
        profit = stake * (odds - 1)

    If horse loses:
        profit = -stake
    """

    bets = bets.copy()

    bets["profit"] = np.where(
        bets[target_col] == 1,
        stake * (bets[odds_col] - 1),
        -stake,
    )

    return bets


def summarize_bets(bets: pd.DataFrame) -> dict:
    """
    Return summary metrics for a group of bets.
    """

    if len(bets) == 0:
        return {
            "bets": 0,
            "profit": 0,
            "roi": 0,
            "hit_rate": 0,
            "avg_odds": 0,
            "max_drawdown": 0,
        }

    bets = bets.copy()
    bets["cumulative_profit"] = bets["profit"].cumsum()
    running_max = bets["cumulative_profit"].cummax()
    drawdown = bets["cumulative_profit"] - running_max

    return {
        "bets": len(bets),
        "profit": bets["profit"].sum(),
        "roi": bets["profit"].mean(),
        "hit_rate": bets["won"].mean(),
        "avg_odds": bets["win_odds"].mean(),
        "max_drawdown": drawdown.min(),
    }


def run_threshold_backtests(
    df: pd.DataFrame,
    thresholds: list[float] | None = None,
    min_odds: float | None = None,
    max_odds: float | None = None,
) -> pd.DataFrame:
    """
    Test multiple expected-profit thresholds.

    Example:
    threshold = 0.20 means only bet when expected_profit > 20%.
    """

    if thresholds is None:
        thresholds = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 2.00]

    results = []

    for threshold in thresholds:
        bets = select_bets(
            df,
            threshold=threshold,
            min_odds=min_odds,
            max_odds=max_odds,
        )

        bets = calculate_bet_profits(bets)
        summary = summarize_bets(bets)

        summary["threshold"] = threshold
        summary["min_odds"] = min_odds
        summary["max_odds"] = max_odds

        results.append(summary)

    results_df = pd.DataFrame(results)

    columns = [
        "threshold",
        "bets",
        "profit",
        "roi",
        "hit_rate",
        "avg_odds",
        "max_drawdown",
        "min_odds",
        "max_odds",
    ]

    return results_df[columns]


def get_strategy_bets(
    df: pd.DataFrame,
    threshold: float = 0.20,
    min_odds: float | None = None,
    max_odds: float | None = None,
    date_col: str = "date",
    race_col: str = "race_id",
) -> pd.DataFrame:
    """
    Return detailed bets for one strategy.
    """

    bets = select_bets(
        df,
        threshold=threshold,
        min_odds=min_odds,
        max_odds=max_odds,
    )

    bets = calculate_bet_profits(bets)

    if date_col in bets.columns:
        bets = bets.sort_values([date_col, race_col]).reset_index(drop=True)
    else:
        bets = bets.sort_values([race_col]).reset_index(drop=True)

    bets["cumulative_profit"] = bets["profit"].cumsum()

    return bets


def plot_cumulative_profit(
    bets: pd.DataFrame,
    title: str = "Cumulative Profit",
) -> None:
    """
    Plot cumulative profit for selected bets.
    """

    if len(bets) == 0:
        print("No bets to plot.")
        return

    bets = bets.reset_index(drop=True).copy()

    if "cumulative_profit" not in bets.columns:
        bets["cumulative_profit"] = bets["profit"].cumsum()

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(bets)), bets["cumulative_profit"])
    plt.xlabel("Bet number")
    plt.ylabel("Cumulative profit")
    plt.title(title)
    plt.show()