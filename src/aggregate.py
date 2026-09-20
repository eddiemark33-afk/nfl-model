"""Stage 2: collapse plays to one row per team per game."""

import pandas as pd


def team_game_table(plays: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game with offensive and defensive EPA/play.

    Columns:
        game_id, season, week, team, opp, home (1 if team was home, else 0),
        off_epa, off_plays, def_epa, def_plays
    """
    off = (
        plays.groupby(["game_id", "season", "week", "posteam", "defteam", "home_team"])
        .agg(off_epa=("epa", "mean"), off_plays=("epa", "size"))
        .reset_index()
        .rename(columns={"posteam": "team", "defteam": "opp"})
    )

    dfn = (
        plays.groupby(["game_id", "defteam"])
        .agg(def_epa=("epa", "mean"), def_plays=("epa", "size"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    tg = off.merge(dfn, on=["game_id", "team"], how="inner")
    tg["home"] = (tg["team"] == tg["home_team"]).astype(int)
    tg = tg.drop(columns=["home_team"])

    cols = ["game_id", "season", "week", "team", "opp", "home",
            "off_epa", "off_plays", "def_epa", "def_plays"]
    return tg[cols].sort_values(["season", "week", "game_id", "home"]).reset_index(drop=True)


if __name__ == "__main__":
    from load import load_plays

    plays = load_plays([2025])
    tg = team_game_table(plays)
    print(f"team-games: {len(tg)} rows ({tg['game_id'].nunique()} games x 2)")
    print(tg.head(6))
    print()
    print("2025 season-to-date, sorted by off_epa - def_epa (unadjusted):")
    season = tg.groupby("team").agg(off_epa=("off_epa", "mean"), def_epa=("def_epa", "mean"))
    season["net"] = season["off_epa"] - season["def_epa"]
    print(season.sort_values("net", ascending=False).round(3).head(8))
    print("DONE")
