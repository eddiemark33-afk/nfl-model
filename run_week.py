"""Entry point: fit current ratings and print this week's card.

Usage:
    python run_week.py            # next week with unplayed games
    python run_week.py --week 5   # a specific week of the current season
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from load import load_plays, load_games          # noqa: E402
from aggregate import team_game_table            # noqa: E402
from ratings import fit_ratings                  # noqa: E402
from spread import walk_forward_diffs, fit_margin_model  # noqa: E402
from backtest import blend_regression            # noqa: E402
from bets import build_card                      # noqa: E402

SEASONS = [2022, 2023, 2024, 2025, 2026]
CURRENT_SEASON = 2026
BACKTEST_START = 2023


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, default=None)
    args = ap.parse_args()

    tg = team_game_table(load_plays(SEASONS))
    games = load_games(SEASONS)
    cur = games[games["season"] == CURRENT_SEASON]

    week = args.week or int(cur[cur["result"].isna()]["week"].min())
    slate = cur[cur["week"] == week]

    ratings = fit_ratings(tg, CURRENT_SEASON, week)
    wf = walk_forward_diffs(tg, games, start_season=BACKTEST_START)
    fit = fit_margin_model(wf)
    beta, hfa = fit.params["rating_diff"], fit.params["const"]
    wf["model_spread"] = beta * wf["rating_diff"] + hfa
    blend = blend_regression(wf)

    card = build_card(slate, ratings, beta, hfa, blend.params)

    out_dir = ROOT / "output" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{CURRENT_SEASON}_wk{week:02d}.csv"
    card.to_csv(out_path, index=False)

    rat_dir = ROOT / "output" / "ratings"
    rat_dir.mkdir(parents=True, exist_ok=True)
    ratings.round(4).to_csv(rat_dir / f"{CURRENT_SEASON}_wk{week:02d}.csv")

    pd.set_option("display.width", 200)
    print(f"\n{CURRENT_SEASON} week {week} card   (beta {beta:.1f}, hfa {hfa:.2f}, "
          f"blend: {blend.params['model_spread']:.2f} model + {blend.params['spread_line']:.2f} market)\n")
    show = ["away", "home", "market_home_line", "model_home_margin", "blend_home_margin",
            "edge_blend", "p_home_cover", "p_home_win", "pick", "stake_pct_bankroll", "fade_candidate"]
    print(card[show].to_string(index=False))
    print(f"\npicks flagged: {(card['pick'] != '').sum()}   fade candidates: {(card['fade_candidate'] != '').sum()}")
    print(f"saved: {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
