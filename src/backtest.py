"""Stage 6: walk-forward backtest against the market.

Two questions:
  1. Does the model carry information the closing line doesn't?
     result ~ model_spread + spread_line  -- model coefficient must be > 0 and significant.
  2. When model and market disagree by X+ points, how often does the model side cover?
"""

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

from spread import walk_forward_diffs, fit_margin_model

OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "backtest"
BREAK_EVEN = 0.5238  # win rate needed at -110


def blend_regression(wf: pd.DataFrame):
    """OLS: result ~ model_spread + spread_line."""
    X = sm.add_constant(wf[["model_spread", "spread_line"]])
    return sm.OLS(wf["result"], X).fit()


def ats_by_threshold(wf: pd.DataFrame, pred_col: str, thresholds=(0.5, 1, 2, 3, 4, 5)) -> pd.DataFrame:
    """Against-the-spread record when betting the side pred_col favors vs the market.

    Bet home when pred > spread_line, away when pred < spread_line.
    Cover for home = result > spread_line. Pushes (result == spread_line) excluded.
    """
    edge = wf[pred_col] - wf["spread_line"]
    home_cover = wf["result"] - wf["spread_line"]
    rows = []
    for t in thresholds:
        sel = wf[(edge.abs() >= t) & (home_cover != 0)]
        e = edge[sel.index]
        hc = home_cover[sel.index]
        wins = ((e > 0) & (hc > 0)) | ((e < 0) & (hc < 0))
        n = len(sel)
        rows.append({
            "min_edge": t,
            "bets": n,
            "wins": int(wins.sum()),
            "win_pct": wins.mean() if n else float("nan"),
            "roi_at_-110": (wins.mean() * 1.909 - 1) if n else float("nan"),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from load import load_plays, load_games
    from aggregate import team_game_table

    seasons = [2022, 2023, 2024, 2025, 2026]
    tg = team_game_table(load_plays(seasons))
    games = load_games(seasons)

    wf = walk_forward_diffs(tg, games, start_season=2023)
    fit = fit_margin_model(wf)
    beta, hfa = fit.params["rating_diff"], fit.params["const"]
    wf["model_spread"] = beta * wf["rating_diff"] + hfa
    print(f"walk-forward games: {len(wf)}   beta {beta:.1f}   hfa {hfa:.2f}")

    # --- Question 1: does the model add information beyond the market? ---
    bl = blend_regression(wf)
    print("\n=== Blend regression: result ~ model_spread + spread_line ===")
    print(f"  model_spread  coef {bl.params['model_spread']:+.3f}   p-value {bl.pvalues['model_spread']:.4f}")
    print(f"  spread_line   coef {bl.params['spread_line']:+.3f}   p-value {bl.pvalues['spread_line']:.4f}")
    print(f"  (coefs sum to ~1 when both are unbiased forecasts; model p < 0.05 means it adds information)")

    wf["blend_spread"] = bl.predict(sm.add_constant(wf[["model_spread", "spread_line"]]))

    # --- Question 2: ATS record when disagreeing with the market ---
    print(f"\n=== ATS record, raw model vs market (break-even {BREAK_EVEN:.1%}) ===")
    raw = ats_by_threshold(wf, "model_spread")
    print(raw.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n=== ATS record, blended prediction vs market ===")
    blend = ats_by_threshold(wf, "blend_spread", thresholds=(0.5, 1, 1.5, 2, 3))
    print(blend.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wf.to_csv(OUT_DIR / "walk_forward_predictions.csv", index=False)
    raw.to_csv(OUT_DIR / "ats_raw_model.csv", index=False)
    blend.to_csv(OUT_DIR / "ats_blended.csv", index=False)
    print(f"\nsaved 3 files to {OUT_DIR}")
    print("DONE")
