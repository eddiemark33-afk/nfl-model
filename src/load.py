"""Stage 1: pull play-by-play and schedule data from nflverse and apply play filters."""

import nflreadpy as nfl
import pandas as pd

from pathlib import Path

# Cache nflverse downloads on disk in data/raw so each season is fetched once.
nfl.config.update_config(
    cache_mode="filesystem",
    cache_dir=Path(__file__).resolve().parent.parent / "data" / "raw",
)

PLAY_COLS = [
    "game_id", "season", "week", "season_type",
    "posteam", "defteam", "home_team", "away_team",
    "play_type", "epa", "success",
]

SCHED_COLS = [
    "game_id", "season", "game_type", "week", "gameday",
    "home_team", "away_team", "home_score", "away_score",
    "result", "spread_line", "total_line",
]


def load_plays(seasons: list[int]) -> pd.DataFrame:
    """Regular-season pass and run plays with a valid EPA, kneels and spikes removed."""
    pbp = nfl.load_pbp(seasons).to_pandas()
    plays = pbp[
        pbp["play_type"].isin(["pass", "run"])
        & pbp["epa"].notna()
        & (pbp["qb_kneel"] == 0)
        & (pbp["qb_spike"] == 0)
        & (pbp["season_type"] == "REG")
    ]
    return plays[PLAY_COLS].reset_index(drop=True)


def load_games(seasons: list[int]) -> pd.DataFrame:
    """Regular-season games with scores and market lines.

    result      = home_score - away_score
    spread_line = home-team spread, positive means home favored
    """
    sched = nfl.load_schedules(seasons).to_pandas()
    games = sched[sched["game_type"] == "REG"]
    return games[SCHED_COLS].reset_index(drop=True)


if __name__ == "__main__":
    plays = load_plays([2025])
    games = load_games([2025])
    print(f"plays: {len(plays):,} rows, {plays['game_id'].nunique()} games")
    print(plays.head())
    print(f"games: {len(games)} rows")
    print(games.head())
    print("DONE")
