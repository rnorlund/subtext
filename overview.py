#!/usr/bin/env python3
"""
overview.py — Cross-relationship comparison across all your contacts.

The single-contact view answers "how is this relationship?". This answers the
bigger question you actually asked: "across everyone, where am I doing well and
where could I do better?" One enriched pass over all messages, aggregated per
1:1 contact into a comparable table.
"""

from __future__ import annotations

import pandas as pd

import signals as sig


def summarize(df: pd.DataFrame, min_msgs: int = 50, top: int = 50) -> pd.DataFrame:
    """Per-contact summary metrics, ranked by volume.

    Runs VADER once over all 1:1 messages, then aggregates — efficient even
    across tens of thousands of messages.
    """
    if df.empty:
        return pd.DataFrame()

    d = sig.enrich(df[~df["is_group"]].copy())
    g = d.groupby("contact")

    pos = g["is_positive"].sum()
    neg = g["is_negative"].sum()

    summary = pd.DataFrame(
        {
            "messages": g.size(),
            "your_share": g["is_from_me"].mean(),
            "net_sentiment": g["compound"].mean(),
            "pct_positive": g["is_positive"].mean(),
            "pct_negative": g["is_negative"].mean(),
            "positivity_ratio": (pos / neg.replace(0, pd.NA)).clip(upper=20),
            "affection": g["affection"].mean(),
            "emoji_rate": g["n_emoji"].mean(),
            "vulnerability": g["vulnerability"].mean(),
            "first": g["dt"].min(),
            "last": g["dt"].max(),
        }
    )
    summary["span_days"] = (summary["last"] - summary["first"]).dt.days
    summary["days_since_last"] = (d["dt"].max() - summary["last"]).dt.days
    summary = summary[summary["messages"] >= min_msgs]
    return summary.sort_values("messages", ascending=False).head(top)


# Column display config for the dashboard table.
DISPLAY_COLS = {
    "messages": "Messages",
    "your_share": "Your share",
    "net_sentiment": "Net sentiment",
    "pct_positive": "% positive",
    "pct_negative": "% negative",
    "positivity_ratio": "Pos:Neg ratio",
    "affection": "Affection",
    "emoji_rate": "Emoji/msg",
    "vulnerability": "Vulnerability",
    "span_days": "Span (days)",
    "days_since_last": "Days since last",
}
