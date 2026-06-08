#!/usr/bin/env python3
"""
trust.py — Conversational signals *related to* trust.

IMPORTANT FRAMING (read before using): trust is built on real-world behavior —
following through, honesty, reliability over time. Text messages capture only a
sliver of that. This module measures LEXICAL PROXIES for trust-relevant talk:

  • commitments    — promises / follow-through language ("I'll", "I promise")
  • distrust        — accusation / suspicion ("you said you'd…", "you forgot",
                      "I don't believe you") — who VOICES distrust, over time
  • accountability  — ownership & repair ("I'm sorry", "my fault", "I was wrong")
  • affirmation     — expressed trust/appreciation ("I trust you", "thank you")

These are signals to REFLECT on and discuss — NOT a measure of whether someone
is trustworthy, and NOT a way to validate or invalidate anyone's feelings.
Treat as directional, never diagnostic.
"""

from __future__ import annotations

import re

import pandas as pd

import signals as sig

CONVERSATION_GAP_MIN = 60

_COMMITMENT = [
    r"\bi'?ll\b", r"i will", r"i promise", r"i swear", r"i got it", r"on my way",
    r"i'?ll take care", r"i'?ll be there", r"i'?ll handle", r"i'?ll do it",
    r"count on me", r"i won'?t forget", r"i'?ll make sure", r"i'?ll get it done",
    r"consider it done", r"i'?ve got it",
]
# Distrust / broken-follow-through / suspicion directed at the other person.
_DISTRUST = [
    r"you said you'?d", r"you said you would", r"you promised", r"you told me you'?d",
    r"you didn'?t", r"you forgot", r"you never", r"you always say", r"where were you",
    r"where are you", r"who is (this|that|she|he)", r"who'?s \w+", r"you lied",
    r"\blying\b", r"that'?s a lie", r"don'?t believe you", r"don'?t trust",
    r"can'?t trust", r"be honest with me", r"tell me the truth", r"are you lying",
    r"are you cheating", r"prove it", r"yeah right", r"sure you did", r"is that true",
    r"why didn'?t you", r"you were supposed to", r"again\b.*you", r"still haven'?t",
]
_AFFIRM = [
    r"i trust you", r"i believe you", r"i know you will", r"i know you can",
    r"thank you for", r"i appreciate you", r"proud of you", r"i'?m grateful",
    r"you'?re reliable", r"you came through", r"i can count on you", r"means a lot",
    r"you'?ve got this", r"i'?m glad i can",
]
_ACCOUNTABILITY = [
    r"i'?m sorry", r"my fault", r"my bad", r"i should have", r"i shouldn'?t have",
    r"i messed up", r"i'?ll do better", r"you'?re right", r"i was wrong",
    r"i apologi", r"that'?s on me", r"i own that", r"i dropped the ball",
]

CATEGORIES = ["commitments", "distrust", "accountability", "affirmation"]
_LEX = {"commitments": _COMMITMENT, "distrust": _DISTRUST,
        "accountability": _ACCOUNTABILITY, "affirmation": _AFFIRM}

LABELS = {
    "commitments": "Commitments made (promises / follow-through language)",
    "distrust": "Distrust voiced (accusation / suspicion / broken-promise)",
    "accountability": "Accountability (ownership, apology, repair)",
    "affirmation": "Trust affirmed (appreciation / expressed trust)",
}


def _count(patterns, text: str) -> int:
    return sum(1 for p in patterns if re.search(p, text))


def enrich_trust(df: pd.DataFrame) -> pd.DataFrame:
    df = sig.enrich(df).sort_values("dt").reset_index(drop=True)
    low = df["text"].fillna("").str.lower()
    for cat, lex in _LEX.items():
        df[cat] = low.apply(lambda t, lx=lex: _count(lx, t))
    return df


def compute_trust(df: pd.DataFrame, freq: str = "ME") -> dict:
    """Per-period COUNTS of each trust category, split by person (+ volume)."""
    if df.empty:
        return {"me": pd.DataFrame(), "them": pd.DataFrame()}
    g = enrich_trust(df)
    out = {}
    for who, mask in (("me", g["is_from_me"]), ("them", ~g["is_from_me"])):
        sub = g[mask].set_index("dt")
        grp = sub.groupby(pd.Grouper(freq=freq))
        d = pd.DataFrame({c: grp[c].sum() for c in CATEGORIES})
        d["volume"] = grp.size()
        out[who] = d[d["volume"] > 0]
    return out


def summary(df: pd.DataFrame) -> dict:
    """Overall per-person rates (per 100 messages) for headline display."""
    if df.empty:
        return {}
    g = enrich_trust(df)
    out = {}
    for who, mask in (("me", g["is_from_me"]), ("them", ~g["is_from_me"])):
        sub = g[mask]
        n = max(len(sub), 1)
        for c in CATEGORIES:
            out[f"{c}_{who}"] = 100 * sub[c].sum() / n
    return out


def flagged_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Every trust-relevant message with its category, for inspection."""
    g = enrich_trust(df)
    rows = g[g[CATEGORIES].sum(axis=1) > 0].copy()
    rows["category"] = rows[CATEGORIES].idxmax(axis=1)
    return rows
