"""Stages 3-4: opponent-adjusted offense/defense ratings via weighted ridge regression.

Model, one row per team-game:
    off_epa = O[team] + D[opp] + H * home + intercept

O[team] = offense rating (higher is better)
D[team] = defense rating, EPA allowed (lower / more negative is better)
net     = O - D
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HALF_LIFE_WEEKS = 10      # time weight halves every 10 game-weeks
OFFSEASON_PENALTY = 8     # extra game-weeks added when crossing a season boundary
DEFAULT_ALPHA = 1.0       # ridge shrinkage


def time_weights(tg: pd.DataFrame, as_of_season: int, as_of_week: int) -> pd.Series:
    """Exponential decay by game-weeks before (as_of_season, as_of_week)."""
    keys = sorted(set(zip(tg["season"], tg["week"])))
    idx = {k: i for i, k in enumerate(keys)}
    idx_as_of = sum(1 for k in keys if k < (as_of_season, as_of_week))

    game_idx = np.array([idx[k] for k in zip(tg["season"], tg["week"])])
    weeks_ago = (idx_as_of - game_idx) + OFFSEASON_PENALTY * (as_of_season - tg["season"].to_numpy())
    return pd.Series(0.5 ** (weeks_ago / HALF_LIFE_WEEKS), index=tg.index)


def fit_ratings(
    tg: pd.DataFrame,
    as_of_season: int,
    as_of_week: int,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Fit ratings using only games played before (as_of_season, as_of_week).

    Returns a DataFrame indexed by team with columns off, def, net, plus the
    fitted home-field term in .attrs["hfa"] (in EPA/play units).
    """
    before = (tg["season"] < as_of_season) | (
        (tg["season"] == as_of_season) & (tg["week"] < as_of_week)
    )
    d = tg[before].copy()
    if d.empty:
        raise ValueError("no games before the requested as_of point")

    teams = sorted(set(d["team"]) | set(d["opp"]))
    t_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    # Design matrix: [offense dummies | defense dummies | home]
    X = np.zeros((len(d), 2 * n + 1))
    X[np.arange(len(d)), d["team"].map(t_idx)] = 1.0
    X[np.arange(len(d)), n + d["opp"].map(t_idx)] = 1.0
    X[:, -1] = d["home"].to_numpy()
    y = d["off_epa"].to_numpy()

    w = d["off_plays"].to_numpy() * time_weights(d, as_of_season, as_of_week).to_numpy()
    w = w / w.mean()

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y, sample_weight=w)

    coef = model.coef_
    out = pd.DataFrame(
        {"off": coef[:n], "def": coef[n : 2 * n]},
        index=pd.Index(teams, name="team"),
    )
    out["net"] = out["off"] - out["def"]
    out.attrs["hfa"] = float(coef[-1])
    out.attrs["intercept"] = float(model.intercept_)
    return out.sort_values("net", ascending=False)


if __name__ == "__main__":
    from load import load_plays
    from aggregate import team_game_table

    tg = team_game_table(load_plays([2022, 2023, 2024, 2025, 2026]))
    print(f"team-games loaded: {len(tg)} rows, seasons {sorted(tg['season'].unique())}")

    r = fit_ratings(tg, as_of_season=2026, as_of_week=3)
    print("\nRatings as of 2026 week 3 (EPA/play):")
    print(r.round(3).to_string())
    print(f"\nhome-field term: {r.attrs['hfa']:.4f} EPA/play")
    print("DONE")
