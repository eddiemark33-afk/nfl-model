"""Stage 5: turn rating differentials into predicted point spreads.

predicted home margin = beta * (net_home - net_away) + hfa_pts

beta and hfa_pts are fit by OLS on walk-forward predictions: for every past
week, ratings are fit using only games before that week, so no game is ever
predicted with information from itself or later.
"""

import pandas as pd
import statsmodels.api as sm

from ratings import fit_ratings


def walk_forward_diffs(
    tg: pd.DataFrame,
    games: pd.DataFrame,
    start_season: int,
    start_week: int = 1,
) -> pd.DataFrame:
    """Rating differential for every completed game from (start_season, start_week) on.

    Returns one row per game: game_id, season, week, home_team, away_team,
    rating_diff, result, spread_line.
    """
    played = games[games["result"].notna()]
    weeks = sorted(
        set(zip(played["season"], played["week"]))
        - {k for k in zip(played["season"], played["week"]) if k < (start_season, start_week)}
    )

    rows = []
    for season, week in weeks:
        r = fit_ratings(tg, season, week)
        net = r["net"]
        g = played[(played["season"] == season) & (played["week"] == week)]
        for _, row in g.iterrows():
            if row["home_team"] not in net.index or row["away_team"] not in net.index:
                continue
            rows.append({
                "game_id": row["game_id"],
                "season": season,
                "week": week,
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "rating_diff": net[row["home_team"]] - net[row["away_team"]],
                "result": row["result"],
                "spread_line": row["spread_line"],
            })
    return pd.DataFrame(rows)


def fit_margin_model(wf: pd.DataFrame):
    """OLS: result ~ rating_diff. Returns fitted statsmodels result."""
    X = sm.add_constant(wf["rating_diff"])
    return sm.OLS(wf["result"], X).fit()


def predict_margin(ratings: pd.DataFrame, home: str, away: str, beta: float, hfa_pts: float) -> float:
    """Predicted home-team margin for one matchup."""
    return beta * (ratings.loc[home, "net"] - ratings.loc[away, "net"]) + hfa_pts


if __name__ == "__main__":
    from load import load_plays, load_games
    from aggregate import team_game_table

    seasons = [2022, 2023, 2024, 2025, 2026]
    tg = team_game_table(load_plays(seasons))
    games = load_games(seasons)

    wf = walk_forward_diffs(tg, games, start_season=2023)
    print(f"walk-forward games: {len(wf)} ({wf['season'].min()}-{wf['season'].max()})")

    fit = fit_margin_model(wf)
    beta, hfa = fit.params["rating_diff"], fit.params["const"]
    print(f"\nbeta (points per 1.0 EPA/play of net rating): {beta:.1f}")
    print(f"home-field advantage: {hfa:.2f} points")
    print(f"R-squared: {fit.rsquared:.3f}   residual std: {fit.resid.std():.2f} points")

    wf["model_spread"] = beta * wf["rating_diff"] + hfa
    mae_model = (wf["result"] - wf["model_spread"]).abs().mean()
    mae_market = (wf["result"] - wf["spread_line"]).abs().mean()
    print(f"\nmean abs error vs actual margin:  model {mae_model:.2f}   market {mae_market:.2f}")
    print("DONE")
