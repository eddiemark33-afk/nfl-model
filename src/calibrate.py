"""Calibration check: are the model's probabilities accurate?

For every walk-forward game (2023 on), compare predicted win probability to what
actually happened. Two measures:
  Brier score  = mean (prob - outcome)^2 ; lower is better; 0.25 = coin flip
  Calibration  = within each probability bucket, did the home team win that often?
Also scored: the book's own vig-free moneyline probability, as the bar to clear.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from spread import walk_forward_diffs, fit_margin_model
from backtest import blend_regression
from bets import win_prob, cover_prob, vig_free_probs

OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "backtest"


def brier(p: pd.Series, y: pd.Series) -> float:
    return float(((p - y) ** 2).mean())


def calibration_table(p: pd.Series, y: pd.Series, bins=(0, .3, .4, .5, .6, .7, .8, 1.0)) -> pd.DataFrame:
    b = pd.cut(p, bins=bins, include_lowest=True)
    t = pd.DataFrame({"bucket": b, "pred": p, "actual": y}).groupby("bucket", observed=True)
    out = t.agg(games=("actual", "size"), predicted=("pred", "mean"), actual=("actual", "mean"))
    out["gap"] = out["actual"] - out["predicted"]
    return out.round(3)


if __name__ == "__main__":
    import statsmodels.api as sm
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
    wf = wf.merge(games[["game_id", "home_moneyline", "away_moneyline"]], on="game_id", how="left")
    wf = wf[wf["result"] != 0].copy()          # drop ties for win/loss scoring

    y_win = (wf["result"] > 0).astype(float)
    p_model = wf["model_spread"].apply(win_prob)
    p_blend = wf["blend_spread"].apply(win_prob)
    has_ml = wf["home_moneyline"].notna() & wf["away_moneyline"].notna()
    p_book = pd.Series(
        [vig_free_probs(h, a)[0] for h, a in zip(wf["home_moneyline"], wf["away_moneyline"])],
        index=wf.index,
    ).where(has_ml)

    print(f"games scored: {len(wf)}   (with book moneyline: {int(has_ml.sum())})\n")
    print("=== WIN PROBABILITY: Brier score (lower is better, 0.250 = coin flip) ===")
    print(f"  raw model      {brier(p_model, y_win):.4f}")
    print(f"  blended model  {brier(p_blend, y_win):.4f}")
    print(f"  book (vig-free){brier(p_book[has_ml], y_win[has_ml]):.4f}")
    print(f"  always 50%     {brier(pd.Series(0.5, index=wf.index), y_win):.4f}")

    print("\n=== CALIBRATION, blended model win prob (predicted vs actual home win rate) ===")
    print(calibration_table(p_blend, y_win).to_string())

    print("\n=== CALIBRATION, book vig-free win prob ===")
    print(calibration_table(p_book[has_ml], y_win[has_ml]).to_string())

    push = wf["result"] == wf["spread_line"]
    y_cov = (wf["result"] > wf["spread_line"]).astype(float)[~push]
    p_cov = pd.Series([cover_prob(m, l) for m, l in zip(wf["blend_spread"], wf["spread_line"])],
                      index=wf.index)[~push]
    print("\n=== COVER PROBABILITY, blended model (book's implied is ~0.500 on every game) ===")
    print(f"  Brier blended {brier(p_cov, y_cov):.4f}   Brier always-50% {brier(pd.Series(0.5, index=p_cov.index), y_cov):.4f}")
    print(calibration_table(p_cov, y_cov, bins=(0, .45, .5, .55, .6, 1.0)).to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"game_id": wf["game_id"], "home_win": y_win, "p_model": p_model,
                  "p_blend": p_blend, "p_book": p_book}).to_csv(OUT_DIR / "calibration_win_prob.csv", index=False)
    print("\nDONE")
