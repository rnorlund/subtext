#!/usr/bin/env python3
"""
dynamics.py — Pursue–withdraw (demand–withdraw) dynamics over time.

The pursuit/withdrawal pattern is one of the most replicated findings in
relationship science (Christensen & Heavey, 1990; Gottman). One partner
"pursues" connection; the other "withdraws." We operationalize pursuit from
message behavior along four interpretable components, each expressed as
*your share* (0..1, where 0.5 == balanced):

    initiation   — share of conversation-openers that are yours
    persistence  — share of "double-texts" (consecutive sends before a reply)
    speed        — share of responsiveness (faster replier pursues)
    expressive   — share of bids for connection (affection words + questions)

A composite **pursuit balance** = mean of the four shares:
    > 0.5  → you are pursuing
    < 0.5  → the other person is pursuing
    = 0.5  → balanced

Each component is shown alongside the composite so the score is transparent,
not a black box — the right posture for a scientifically-credible product.
"""

from __future__ import annotations

import pandas as pd

import signals as sig

CONVERSATION_GAP_MIN = 60
MAX_REPLY_MIN = 24 * 60  # ignore gaps longer than a day as "replies"


def _safe_share(mine: float, theirs: float) -> float:
    """My share of a quantity; 0.5 when there's nothing to compare."""
    total = mine + theirs
    return 0.5 if total == 0 else mine / total


def compute_pursuit(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Return a time-indexed DataFrame of pursuit components + composite.

    Convention: 'me' = is_from_me True. Shares are *your* share, so >0.5 means
    you exhibit more of that pursuit behavior than the other person.
    """
    if df.empty:
        return pd.DataFrame()

    df = sig.enrich(df).sort_values("dt").reset_index(drop=True)
    df["prev_from_me"] = df["is_from_me"].shift()
    df["gap_min"] = df["dt"].diff().dt.total_seconds() / 60.0

    # Conversation openers (first message after >gap silence).
    df["is_opener"] = (df["gap_min"] > CONVERSATION_GAP_MIN) | df["gap_min"].isna()

    # Double-text: a message from the same sender as the previous one (no reply
    # in between) and within a plausible window — i.e. persistence.
    df["is_consecutive"] = (
        (df["is_from_me"] == df["prev_from_me"])
        & (df["gap_min"] <= MAX_REPLY_MIN)
    )

    # Replies: sender flipped — latency belongs to the responder.
    df["is_reply"] = (
        (df["is_from_me"] != df["prev_from_me"])
        & df["prev_from_me"].notna()
        & (df["gap_min"] <= MAX_REPLY_MIN)
    )
    # "Bids for connection" — affection + questions.
    df["bids"] = df["affection"] + df["has_question"].astype(int)

    rows = []
    for period, gdf in df.groupby(pd.Grouper(key="dt", freq=freq)):
        if gdf.empty:
            continue

        # initiation share
        openers = gdf[gdf["is_opener"]]
        init = _safe_share(
            (openers["is_from_me"]).sum(), (~openers["is_from_me"]).sum()
        )

        # persistence share (double-texting)
        cons = gdf[gdf["is_consecutive"]]
        persist = _safe_share(
            (cons["is_from_me"]).sum(), (~cons["is_from_me"]).sum()
        )

        # responsiveness share — faster median reply = more pursuit.
        replies = gdf[gdf["is_reply"]]
        my_lat = replies.loc[replies["is_from_me"], "gap_min"].median()
        her_lat = replies.loc[~replies["is_from_me"], "gap_min"].median()
        if pd.isna(my_lat) and pd.isna(her_lat):
            speed = 0.5
        else:
            my_speed = 0.0 if pd.isna(my_lat) else 1.0 / (my_lat + 1)
            her_speed = 0.0 if pd.isna(her_lat) else 1.0 / (her_lat + 1)
            speed = _safe_share(my_speed, her_speed)

        # expressiveness share — bids for connection.
        expr = _safe_share(
            gdf.loc[gdf["is_from_me"], "bids"].sum(),
            gdf.loc[~gdf["is_from_me"], "bids"].sum(),
        )

        composite = (init + persist + speed + expr) / 4.0
        rows.append(
            {
                "dt": period,
                "initiation": init,
                "persistence": persist,
                "speed": speed,
                "expressive": expr,
                "pursuit_balance": composite,
                "volume": len(gdf),
            }
        )

    out = pd.DataFrame(rows).set_index("dt")
    return out[out["volume"] > 0]


COMPONENT_LABELS = {
    "initiation": "Initiation (who opens)",
    "persistence": "Persistence (double-texting)",
    "speed": "Responsiveness (reply speed)",
    "expressive": "Expressiveness (bids: affection + questions)",
    "pursuit_balance": "Pursuit balance (composite)",
}


def _norm01(s: pd.Series) -> pd.Series:
    """Min-max normalize to 0..1; flat series -> 0.5."""
    s = s.astype(float)
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng and rng > 0 else s * 0 + 0.5


def compute_push_pull(df: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    """Per-person ABSOLUTE pursuit & withdrawal levels (0..1) over time.

    Unlike pursuit_balance (a zero-sum share), these are independent levels so
    you can plot one person's pursuit against the other's withdrawal and see
    push–pull patterns (you push ↑ while they pull back ↑).

    Pursuit  = initiation + persistence(double-texting) + responsiveness + bids
    Withdrawal = slow replies + terseness + low initiation
    """
    if df.empty:
        return pd.DataFrame()

    df = sig.enrich(df).sort_values("dt").reset_index(drop=True)
    prev = df["is_from_me"].shift()
    gap = df["dt"].diff().dt.total_seconds() / 60.0
    df["is_reply"] = (df["is_from_me"] != prev) & prev.notna() & (gap <= 24 * 60)
    df["reply_lat"] = gap.where(df["is_reply"])
    df["is_consecutive"] = (df["is_from_me"] == prev) & (gap <= 24 * 60)
    df["is_opener"] = (gap > CONVERSATION_GAP_MIN) | gap.isna()
    df["bids"] = df["affection"] + df["has_question"].astype(int)

    rows = []
    for period, gd in df.groupby(pd.Grouper(key="dt", freq=freq)):
        if gd.empty:
            continue
        rec = {"dt": period}
        for who, m in (("me", gd["is_from_me"]), ("them", ~gd["is_from_me"])):
            sub = gd[m]
            n = max(len(sub), 1)
            lat = sub["reply_lat"].median()
            rec[f"{who}_init"] = sub["is_opener"].sum() / n
            rec[f"{who}_persist"] = sub["is_consecutive"].sum() / n
            rec[f"{who}_resp"] = 1.0 / (lat + 1) if pd.notna(lat) else 0.0
            rec[f"{who}_bids"] = sub["bids"].sum() / n
            rec[f"{who}_lat"] = lat
            rec[f"{who}_words"] = sub["n_words"].mean()
        rows.append(rec)

    d = pd.DataFrame(rows).set_index("dt")
    for who in ("me", "them"):
        d[f"{who}_pursuit"] = (
            _norm01(d[f"{who}_init"]) + _norm01(d[f"{who}_persist"])
            + _norm01(d[f"{who}_resp"]) + _norm01(d[f"{who}_bids"])
        ) / 4.0
        lat = d[f"{who}_lat"]
        lat = lat.fillna(lat.max())
        d[f"{who}_withdrawal"] = (
            _norm01(lat) + (1 - _norm01(d[f"{who}_words"])) + (1 - _norm01(d[f"{who}_init"]))
        ) / 3.0

    # Each person's stance: +leaning in (pursuing), −pulling back (withdrawing).
    d["me_stance"] = d["me_pursuit"] - d["me_withdrawal"]
    d["them_stance"] = d["them_pursuit"] - d["them_withdrawal"]
    # Push–pull POLARIZATION (the signed index that actually shows the pattern):
    #   > 0  → you lean in while they pull back  (you push / they pull)
    #   < 0  → they lean in while you pull back   (they push / you pull)
    #   ≈ 0  → symmetric (both engaged OR both withdrawn)
    d["polarization"] = d["me_stance"] - d["them_stance"]
    # Total engagement disambiguates ≈0 (both-in vs both-out).
    d["engagement"] = (d["me_pursuit"] + d["them_pursuit"]) / 2.0
    return d[[
        "me_pursuit", "them_pursuit", "me_withdrawal", "them_withdrawal",
        "me_stance", "them_stance", "polarization", "engagement",
    ]]


def who_leads(df: pd.DataFrame, freq: str = "ME") -> dict:
    """Emotional-leadership / contagion analysis: who drives the mood?

    Message-level: when one person sends an emotional message, how often does
    the other RECIPROCATE in their next reply? If her replies catch your
    negativity more than yours catch hers, *you lead* negativity. Whoever is
    reciprocated-toward more is the leader; the follower mirrors them.

    Also returns a lag cross-correlation of per-period sentiment.
    """
    if df.empty:
        return {}

    g = sig.enrich(df).sort_values("dt").reset_index(drop=True)
    # Contagion: pairs where sender flips (a reply).
    cont = {"neg": {"me_to_them": [0, 0], "them_to_me": [0, 0]},
            "pos": {"me_to_them": [0, 0], "them_to_me": [0, 0]}}
    fm = g["is_from_me"].values
    posv = g["is_positive"].values
    negv = g["is_negative"].values
    for i in range(1, len(g)):
        if fm[i] == fm[i - 1]:
            continue  # same sender, not a reply
        direction = "me_to_them" if fm[i - 1] else "them_to_me"
        for val, arr in (("neg", negv), ("pos", posv)):
            if arr[i - 1]:  # initiator was emotional
                cont[val][direction][1] += 1            # opportunities
                if arr[i]:                              # responder reciprocated
                    cont[val][direction][0] += 1

    def prob(pair):
        hit, tot = pair
        return hit / tot if tot else 0.0

    neg_me_them = prob(cont["neg"]["me_to_them"])   # she catches my negativity
    neg_them_me = prob(cont["neg"]["them_to_me"])   # I catch her negativity
    pos_me_them = prob(cont["pos"]["me_to_them"])
    pos_them_me = prob(cont["pos"]["them_to_me"])

    # Lag cross-correlation of period-level sentiment.
    gi = g.set_index("dt").groupby(pd.Grouper(freq=freq))
    me_s = gi.apply(lambda x: x.loc[x["is_from_me"], "compound"].mean())
    them_s = gi.apply(lambda x: x.loc[~x["is_from_me"], "compound"].mean())
    pair = pd.concat([me_s.rename("me"), them_s.rename("them")], axis=1).dropna()
    crosscorr = {}
    for lag in range(-3, 4):
        if len(pair) > abs(lag) + 2:
            crosscorr[lag] = float(pair["me"].corr(pair["them"].shift(lag)))

    def verdict(me_to_them, them_to_me, kind):
        if abs(me_to_them - them_to_me) < 0.03:
            return f"{kind}: roughly mutual"
        leader = "You lead" if me_to_them > them_to_me else "They lead"
        return f"{kind}: {leader} (reciprocated {max(me_to_them, them_to_me):.0%} of the time)"

    return {
        "neg_me_to_them": neg_me_them, "neg_them_to_me": neg_them_me,
        "pos_me_to_them": pos_me_them, "pos_them_to_me": pos_them_me,
        "neg_verdict": verdict(neg_me_them, neg_them_me, "Negativity"),
        "pos_verdict": verdict(pos_me_them, pos_them_me, "Positivity"),
        "crosscorr": crosscorr,
    }
