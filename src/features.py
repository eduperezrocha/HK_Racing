# src/features.py

from pathlib import Path
import pandas as pd


BASE_FEATURES = [
    # Horse-level pre-race features
    "horse_age",
    "horse_country",
    "horse_type",
    "horse_rating",
    "horse_gear",
    "declared_weight",
    "actual_weight",
    "draw",
    "trainer_id",
    "jockey_id",

    # Race-level pre-race features
    "venue",
    "config",
    "surface",
    "distance",
    "going",
    "horse_ratings",
    "prize",
    "race_class",
]


HISTORICAL_FEATURES = [
    # Horse lifetime history
    "horse_prev_runs",
    "horse_prev_win_rate",
    "horse_prev_avg_result",
    "horse_prev_avg_odds",
    "horse_prev_best_result",
    "days_since_last_race",

    # Recent horse form
    "horse_last_3_avg_result",
    "horse_last_5_avg_result",
    "horse_last_3_win_rate",
    "horse_last_5_win_rate",
    "horse_last_3_avg_odds",
    "horse_last_5_avg_odds",

    # Distance history
    "horse_distance_prev_runs",
    "horse_distance_prev_win_rate",
    "horse_distance_prev_avg_result",

    # Venue history
    "horse_venue_prev_runs",
    "horse_venue_prev_win_rate",
    "horse_venue_prev_avg_result",

    # Jockey history
    "jockey_prev_runs",
    "jockey_prev_win_rate",
    "jockey_prev_avg_result",

    # Trainer history
    "trainer_prev_runs",
    "trainer_prev_win_rate",
    "trainer_prev_avg_result",
]


def load_raw_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load races.csv and runs.csv.
    """

    data_dir = Path(data_dir)

    races = pd.read_csv(data_dir / "races.csv")
    runs = pd.read_csv(data_dir / "runs.csv")

    return races, runs


def prepare_base_dataframe(
    races: pd.DataFrame,
    runs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge runs and races, create target column, and clean basic types.
    """

    df = runs.merge(races, on="race_id", how="left")

    df["won"] = (df["result"] == 1).astype(int)
    df["date"] = pd.to_datetime(df["date"])

    numeric_cols = [
        "result",
        "win_odds",
        "place_odds",
        "horse_age",
        "horse_rating",
        "declared_weight",
        "actual_weight",
        "draw",
        "distance",
        "prize",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    id_cols = [
        "horse_id",
        "jockey_id",
        "trainer_id",
        "horse_country",
        "horse_type",
        "horse_gear",
        "venue",
        "config",
        "surface",
        "going",
        "horse_ratings",
        "race_class",
    ]

    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)

    return df


def add_horse_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add historical features for each horse.

    Uses shift() so today's result is not used to predict today's race.
    """

    df = df.sort_values(["horse_id", "date", "race_id"]).copy()

    df["horse_prev_runs"] = df.groupby("horse_id").cumcount()

    df["horse_prev_win_rate"] = (
        df.groupby("horse_id")["won"]
        .transform(lambda x: x.shift().expanding().mean())
    )

    df["horse_prev_avg_result"] = (
        df.groupby("horse_id")["result"]
        .transform(lambda x: x.shift().expanding().mean())
    )

    df["horse_prev_avg_odds"] = (
        df.groupby("horse_id")["win_odds"]
        .transform(lambda x: x.shift().expanding().mean())
    )

    df["horse_prev_best_result"] = (
        df.groupby("horse_id")["result"]
        .transform(lambda x: x.shift().expanding().min())
    )

    df["prev_race_date"] = df.groupby("horse_id")["date"].shift()

    df["days_since_last_race"] = (
        df["date"] - df["prev_race_date"]
    ).dt.days

    return df


def add_recent_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling recent-form features for each horse.
    """

    df = df.sort_values(["horse_id", "date", "race_id"]).copy()

    df["horse_last_3_avg_result"] = (
        df.groupby("horse_id")["result"]
        .transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
    )

    df["horse_last_5_avg_result"] = (
        df.groupby("horse_id")["result"]
        .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    )

    df["horse_last_3_win_rate"] = (
        df.groupby("horse_id")["won"]
        .transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
    )

    df["horse_last_5_win_rate"] = (
        df.groupby("horse_id")["won"]
        .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    )

    df["horse_last_3_avg_odds"] = (
        df.groupby("horse_id")["win_odds"]
        .transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
    )

    df["horse_last_5_avg_odds"] = (
        df.groupby("horse_id")["win_odds"]
        .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    )

    return df


def add_distance_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add horse history at the same distance.
    """

    if "distance" not in df.columns:
        return df

    df = df.sort_values(["horse_id", "distance", "date", "race_id"]).copy()

    group_cols = ["horse_id", "distance"]

    df["horse_distance_prev_runs"] = df.groupby(group_cols).cumcount()

    df["horse_distance_prev_win_rate"] = (
        df.groupby(group_cols)["won"]
        .transform(lambda x: x.shift().expanding().mean())
    )

    df["horse_distance_prev_avg_result"] = (
        df.groupby(group_cols)["result"]
        .transform(lambda x: x.shift().expanding().mean())
    )

    return df


def add_venue_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add horse history at the same venue.
    """

    if "venue" not in df.columns:
        return df

    df = df.sort_values(["horse_id", "venue", "date", "race_id"]).copy()

    group_cols = ["horse_id", "venue"]

    df["horse_venue_prev_runs"] = df.groupby(group_cols).cumcount()

    df["horse_venue_prev_win_rate"] = (
        df.groupby(group_cols)["won"]
        .transform(lambda x: x.shift().expanding().mean())
    )

    df["horse_venue_prev_avg_result"] = (
        df.groupby(group_cols)["result"]
        .transform(lambda x: x.shift().expanding().mean())
    )

    return df


def add_entity_history_features(
    df: pd.DataFrame,
    entity_col: str,
    prefix: str,
) -> pd.DataFrame:
    """
    Add historical features for jockeys or trainers.

    This avoids same-race leakage by aggregating at entity/date/race_id level.
    """

    if entity_col not in df.columns:
        return df

    race_stats = (
        df.groupby([entity_col, "date", "race_id"], as_index=False)
        .agg(
            entity_runs_in_race=("won", "size"),
            entity_wins_in_race=("won", "sum"),
            entity_result_sum_in_race=("result", "sum"),
        )
    )

    race_stats = race_stats.sort_values([entity_col, "date", "race_id"]).copy()

    group = race_stats.groupby(entity_col)

    race_stats[f"{prefix}_prev_runs"] = (
        group["entity_runs_in_race"].cumsum()
        - race_stats["entity_runs_in_race"]
    )

    race_stats[f"{prefix}_prev_wins"] = (
        group["entity_wins_in_race"].cumsum()
        - race_stats["entity_wins_in_race"]
    )

    previous_result_sum = (
        group["entity_result_sum_in_race"].cumsum()
        - race_stats["entity_result_sum_in_race"]
    )

    race_stats[f"{prefix}_prev_win_rate"] = (
        race_stats[f"{prefix}_prev_wins"] / race_stats[f"{prefix}_prev_runs"]
    )

    race_stats[f"{prefix}_prev_avg_result"] = (
        previous_result_sum / race_stats[f"{prefix}_prev_runs"]
    )

    merge_cols = [
        entity_col,
        "date",
        "race_id",
        f"{prefix}_prev_runs",
        f"{prefix}_prev_win_rate",
        f"{prefix}_prev_avg_result",
    ]

    df = df.merge(
        race_stats[merge_cols],
        on=[entity_col, "date", "race_id"],
        how="left",
    )

    return df


def build_feature_dataset(data_dir: str | Path) -> pd.DataFrame:
    """
    Complete feature-building pipeline.
    """

    races, runs = load_raw_data(data_dir)
    df = prepare_base_dataframe(races, runs)

    df = add_horse_history_features(df)
    df = add_recent_form_features(df)
    df = add_distance_history_features(df)
    df = add_venue_history_features(df)

    df = add_entity_history_features(df, "jockey_id", "jockey")
    df = add_entity_history_features(df, "trainer_id", "trainer")

    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return the list of features that exist in the dataframe.
    """

    features = BASE_FEATURES + HISTORICAL_FEATURES

    features = [col for col in features if col in df.columns]

    # Remove duplicates while preserving order
    features = list(dict.fromkeys(features))

    return features