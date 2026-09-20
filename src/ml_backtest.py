"""Moneyline backtest: bet whenever the model's win probability beats the book's price.

For each walk-forward game (2023 on), compute the blended model's home win prob.
For each side, edge = model prob - implied prob at the actual odds (vig included).
Bet the side with edge >= threshold, flat 1 unit. Report record, ROI, and by season.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from spread import walk_forward_diffs, fit_margin_model
from backtest import blend_regression
from bets import win_prob, implied_prob, american_to_decimal

OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "backtest"
THRESHOLDS = (0.00, 0.01, 0.02, 0.03, 0.05)


def ml_bets(wf: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One row per bet placed under the rule at this threshold."""
    rows = []
    for _, g in wf.iterrows():
        p_h = g["p_home_win"]
        edge_h = p_h - implied_prob(g["home_moneyline"])
        edge_a = (1 - p_h) - implied_prob(g["away_moneyline"])
        if max(edge_h, edge_a) < threshold:
            continue
        if edge_h >= edge_a:
            side, odds, edge, won = g["home_team"], g["home_moneyline"], edge_h, g["result"] > 0
        else:
            side, odds, edge, won = g["away_team"], g["away_moneyline"], edge_a, g["result"] < 0
        profit = (american_to_decimal(odds) - 1) if won else -1.0
        rows.append({"game_id": g["game_id"], "season": g["season"], "week": g["week"],
                     "side": side, "odds": odds, "edge": edge, "won": bool(won), "profit": profit})
    return pd.DataFrame(rows)


def summarize(b: pd.DataFrame) -> dict:
    if b.empty:
        return {"bets": 0, "wins": 0, "win_pct": np.nan, "units": 0.0, "roi": np.nan,
                "favorites": 0, "underdogs": 0}
    return {"bets": len(b), "wins": int(b["won"].sum()), "win_pct": b["won"].mean(),
            "units": b["profit"].sum(), "roi": b["profit"].mean(),
            "favorites": int((b["odds"] < 0).sum()), "underdogs": int((b["odds"] > 0).sum())}


if __name__ == "__main__":
    from load import load_plays, load_games
    from aggregate import team_game_table

    seasons = [2022, 2023, 2024, 2025, 2026]
    tg = team_game_table(load_plays(seasons))
    games = load_games(seasons)

    wf = walk_forward_diffs(tg, games, start_season=2023)
    fit = fit_margin_model(wf)
    wf["model_spread"] = fit.params["rating_diff"] * wf["rating_diff"] + fit.params["const"]
    bl = blend_regression(wf)
    wf["blend_spread"] = bl.predict(sm.add_constant(wf[["model_spread", "spread_line"]]))
    wf["p_home_win"] = wf["blend_spread"].apply(win_prob)
    wf = wf.merge(games[["game_id", "home_moneyline", "away_moneyline"]], on="game_id", how="left")
    wf = wf[(wf["result"] != 0) & wf["home_moneyline"].notna() & wf["away_moneyline"].notna()].copy()
    print(f"games available: {len(wf)}\n")

    print("=== Flat 1-unit bets, by minimum edge (model prob minus book implied prob) ===")
    rows = []
    for t in THRESHOLDS:
        s = summarize(ml_bets(wf, t))
        s["min_edge"] = t
        rows.append(s)
    tbl = pd.DataFrame(rows)[["min_edge", "bets", "wins", "win_pct", "units", "roi", "favorites", "underdogs"]]
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== By season at min edge 0.02 ===")
    b = ml_bets(wf, 0.02)
    by = b.groupby("season").apply(lambda d: pd.Series(summarize(d)), include_groups=False)
    print(by.to_string(float_format=lambda x: f"{x:.3f}"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    b.to_csv(OUT_DIR / "ml_bets_edge02.csv", index=False)
    tbl.to_csv(OUT_DIR / "ml_backtest_summary.csv", index=False)
    print("\nDONE")
