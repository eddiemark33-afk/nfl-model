"""Stage 7: edge, probabilities, and stake sizing for one week's card."""

import numpy as np
import pandas as pd
from scipy.stats import norm

MARGIN_SD = 13.3          # residual std of actual margin around prediction (from stage 5)
KELLY_FRACTION = 0.25     # quarter Kelly
BLEND_MIN_EDGE = 1.5      # points of blend-vs-market disagreement to flag a pick
FADE_MIN_EDGE = 3.0       # points of raw-model-vs-market disagreement to flag a fade candidate
ML_MIN_EDGE = 0.02        # model win prob minus book implied prob (at actual odds) to flag a moneyline pick


def american_to_decimal(odds: float) -> float:
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))


def implied_prob(odds: float) -> float:
    """Raw implied probability of American odds (includes vig)."""
    return 1 / american_to_decimal(odds)


def vig_free_probs(home_odds: float, away_odds: float) -> tuple[float, float]:
    """Implied win probabilities with the vig removed (normalized to sum to 1)."""
    h, a = implied_prob(home_odds), implied_prob(away_odds)
    return h / (h + a), a / (h + a)


def cover_prob(pred_margin: float, line: float, sd: float = MARGIN_SD) -> float:
    """P(home covers) = P(actual margin > line) under Normal(pred_margin, sd)."""
    return float(1 - norm.cdf(line, loc=pred_margin, scale=sd))


def win_prob(pred_margin: float, sd: float = MARGIN_SD) -> float:
    """P(home wins outright)."""
    return cover_prob(pred_margin, 0.0, sd)


def kelly_fraction(p: float, odds: float = -110, fraction: float = KELLY_FRACTION) -> float:
    """Fraction of bankroll to stake. Zero if no edge."""
    b = american_to_decimal(odds) - 1
    f = (p * b - (1 - p)) / b
    return max(0.0, f * fraction)


def build_card(games: pd.DataFrame, ratings: pd.DataFrame, beta: float, hfa: float,
               blend_params: pd.Series) -> pd.DataFrame:
    """One row per game with model, blend, market, edges, probabilities, and flags."""
    rows = []
    for _, g in games.iterrows():
        h, a = g["home_team"], g["away_team"]
        if h not in ratings.index or a not in ratings.index:
            continue
        model = beta * (ratings.loc[h, "net"] - ratings.loc[a, "net"]) + hfa
        market = g["spread_line"]

        if pd.isna(market):
            # No market line in nflverse yet: report the model margin only.
            rows.append({
                "game_id": g["game_id"], "week": g["week"], "gameday": g["gameday"],
                "away": a, "home": h, "market_home_line": np.nan,
                "model_home_margin": round(model, 1), "blend_home_margin": np.nan,
                "edge_raw": np.nan, "edge_blend": np.nan, "p_home_cover": np.nan,
                "p_home_win": round(win_prob(model), 3), "pick": "", "pick_cover_prob": np.nan,
                "stake_pct_bankroll": 0.0, "fade_candidate": "",
                "home_ml": np.nan, "away_ml": np.nan, "book_p_home_win": np.nan,
                "ml_pick": "", "ml_edge": np.nan, "ml_stake_pct_bankroll": 0.0,
            })
            continue

        blend = blend_params["const"] + blend_params["model_spread"] * model + blend_params["spread_line"] * market

        edge_raw = model - market
        edge_blend = blend - market
        p_home_cover = cover_prob(blend, market)

        if abs(edge_blend) >= BLEND_MIN_EDGE:
            side = h if edge_blend > 0 else a
            p = p_home_cover if edge_blend > 0 else 1 - p_home_cover
            pick, pick_p, stake = side, p, kelly_fraction(p)
        else:
            pick, pick_p, stake = "", np.nan, 0.0

        fade = ""
        if abs(edge_raw) >= FADE_MIN_EDGE:
            fade = h if edge_raw < 0 else a   # market side, against the raw model

        # Moneyline: model win prob vs the book's implied prob at the actual odds (vig included).
        p_home_win = win_prob(blend)
        home_ml, away_ml = g["home_moneyline"], g["away_moneyline"]
        ml_pick, ml_stake, ml_edge, book_p_home = "", 0.0, np.nan, np.nan
        if not (pd.isna(home_ml) or pd.isna(away_ml)):
            book_p_home = vig_free_probs(home_ml, away_ml)[0]
            edge_h = p_home_win - implied_prob(home_ml)
            edge_a = (1 - p_home_win) - implied_prob(away_ml)
            if max(edge_h, edge_a) >= ML_MIN_EDGE:
                if edge_h >= edge_a:
                    ml_pick, ml_edge, ml_stake = h, edge_h, kelly_fraction(p_home_win, home_ml)
                else:
                    ml_pick, ml_edge, ml_stake = a, edge_a, kelly_fraction(1 - p_home_win, away_ml)

        rows.append({
            "game_id": g["game_id"], "week": g["week"], "gameday": g["gameday"],
            "away": a, "home": h,
            "market_home_line": market,
            "model_home_margin": round(model, 1),
            "blend_home_margin": round(blend, 1),
            "edge_raw": round(edge_raw, 1),
            "edge_blend": round(edge_blend, 1),
            "p_home_cover": round(p_home_cover, 3),
            "p_home_win": round(p_home_win, 3),
            "pick": pick, "pick_cover_prob": round(pick_p, 3) if pick else np.nan,
            "stake_pct_bankroll": round(stake * 100, 2),
            "fade_candidate": fade,
            "home_ml": home_ml, "away_ml": away_ml,
            "book_p_home_win": round(book_p_home, 3) if not pd.isna(book_p_home) else np.nan,
            "ml_pick": ml_pick,
            "ml_edge": round(ml_edge, 3) if not pd.isna(ml_edge) else np.nan,
            "ml_stake_pct_bankroll": round(ml_stake * 100, 2),
        })
    return pd.DataFrame(rows)
