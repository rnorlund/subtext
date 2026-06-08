#!/usr/bin/env python3
"""
gottman.py — Gottman-method relationship metrics from message text.

Implements text proxies for the major constructs in Gottman's research:

  • The Four Horsemen (predictors of relationship breakdown):
      - Criticism      — attacking character ("you always", "you never")
      - Contempt       — disrespect/mockery (THE strongest divorce predictor)
      - Defensiveness  — counter-blame, denying responsibility
      - Stonewalling   — shutting down / dismissive withdrawal
  • Repair attempts   — de-escalation ("I'm sorry", "you're right", "let's…")
  • Harsh startup     — conversations that OPEN negatively
  • Positivity ratio  — positive ÷ negative (the 5:1 "magic ratio")

These are lexical proxies, not the trained observational coding system Gottman
used in the lab. They are directional signals for a scientifically-honest tool,
NOT clinical diagnoses — label them as such in any UI.
"""

from __future__ import annotations

import re

import pandas as pd

import signals as sig

CONVERSATION_GAP_MIN = 60

# ── Four Horsemen lexicons (lowercased; phrase + regex matching) ───────────
_CRITICISM = [
    r"you always", r"you never", r"why do you always", r"why can'?t you",
    r"you should", r"you shouldn'?t", r"what'?s wrong with you", r"you'?re so",
    r"you don'?t ever", r"you can'?t even", r"typical of you", r"you'?re always",
    r"you make me", r"all you do", r"you don'?t care", r"you don'?t listen",
    r"you forgot", r"you didn'?t even", r"how could you", r"why would you",
    r"you really", r"you have to", r"you need to", r"every time you",
    r"you keep", r"you'?re not even",
]
_CONTEMPT = [
    r"whatever", r"\bstupid\b", r"\bidiot\b", r"\bdumb\b", r"\bloser\b",
    r"pathetic", r"ridiculous", r"\bgrow up\b", r"you'?re such a", r"shut up",
    r"good for you", r"🙄", r"\bduh\b", r"real mature", r"\bselfish\b", r"\blazy\b",
    r"\bjerk\b", r"\bannoying\b", r"unbelievable", r"are you kidding",
    r"are you serious", r"wow\.", r"\bnice job\b", r"so smart", r"figures",
    r"of course you", r"\bridiculous\b", r"\bpitiful\b",
]
_DEFENSIVENESS = [
    r"not my fault", r"it'?s not my", r"i didn'?t do", r"why are you blaming",
    r"yeah but", r"well you", r"that'?s not true", r"i never said", r"don'?t blame me",
    r"you'?re the one", r"i was just", r"stop attacking", r"i already",
    r"i told you", r"you didn'?t tell me", r"that'?s not what i", r"i can'?t help it",
    r"you'?re overreacting", r"calm down", r"i was going to", r"don'?t yell",
    r"i never did", r"that'?s not fair", r"why is it always my",
]
# Tightened: drop benign "ok/okay/sure/cool/done" (huge false-positive sources).
_STONEWALL = [
    r"^k$", r"^kk$", r"^whatever$", r"^nvm$", r"^idc$", r"don'?t want to talk",
    r"i'?m done", r"leave me alone", r"forget it", r"not talking about this",
    r"i don'?t care anymore", r"stop talking", r"^done\.$", r"end of discussion",
]
_REPAIR = [
    r"i'?m sorry", r"i apologi", r"my bad", r"you'?re right", r"i understand",
    r"let'?s", r"can we", r"i hear you", r"i love you", r"fair enough",
    r"i didn'?t mean", r"let me try", r"i appreciate", r"thank you", r"i get it",
]


def _count(patterns, text: str) -> int:
    return sum(1 for p in patterns if re.search(p, text))


def enrich_gottman(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-message Gottman flags."""
    df = sig.enrich(df).sort_values("dt").reset_index(drop=True)
    low = df["text"].fillna("").str.lower().str.strip()
    df["criticism"] = low.apply(lambda t: _count(_CRITICISM, t))
    df["contempt"] = low.apply(lambda t: _count(_CONTEMPT, t))
    df["defensiveness"] = low.apply(lambda t: _count(_DEFENSIVENESS, t))
    df["stonewalling"] = low.apply(lambda t: _count(_STONEWALL, t))
    df["repair"] = low.apply(lambda t: _count(_REPAIR, t))
    # Harsh startup: a conversation-opener that is negative in tone.
    gap = df["dt"].diff() > pd.Timedelta(minutes=CONVERSATION_GAP_MIN)
    df["is_opener"] = gap.fillna(True)
    df["harsh_startup"] = (df["is_opener"] & (df["compound"] <= -0.05)).astype(int)
    return df


HORSEMEN = ["criticism", "contempt", "defensiveness", "stonewalling"]


def compute_gottman(df: pd.DataFrame, freq: str = "ME") -> dict:
    """Per-period Gottman metrics, split by person.

    Returns {"me": DataFrame, "them": DataFrame, "shared": DataFrame} where
    shared holds pair-level series (positivity_ratio, harsh_startup_rate).
    """
    if df.empty:
        return {"me": pd.DataFrame(), "them": pd.DataFrame(), "shared": pd.DataFrame()}

    g = enrich_gottman(df)
    cols = HORSEMEN + ["repair"]

    out = {}
    for who, mask in (("me", g["is_from_me"]), ("them", ~g["is_from_me"])):
        sub = g[mask].set_index("dt")
        grp = sub.groupby(pd.Grouper(freq=freq))
        # COUNTS per period (clearer than tiny per-message rates for rare events).
        d = pd.DataFrame({c: grp[c].sum() for c in cols})
        d["volume"] = grp.size()
        out[who] = d[d["volume"] > 0]

    # Pair-level.
    gi = g.set_index("dt").groupby(pd.Grouper(freq=freq))
    pos = gi["is_positive"].sum()
    neg = gi["is_negative"].sum()
    shared = pd.DataFrame({
        "positivity_ratio": (pos / neg.replace(0, pd.NA)).clip(upper=20),
        "harsh_startup_rate": gi["harsh_startup"].mean(),
        "volume": gi.size(),
    })
    out["shared"] = shared[shared["volume"] > 0]
    return out


def summary(df: pd.DataFrame) -> dict:
    """Overall Gottman snapshot for headline display."""
    if df.empty:
        return {}
    g = enrich_gottman(df)
    n = len(g)
    pos = int(g["is_positive"].sum())
    neg = int(g["is_negative"].sum())
    out = {
        "positivity_ratio": pos / neg if neg else float("inf"),
        "harsh_startup_pct": 100 * g.loc[g["is_opener"], "harsh_startup"].mean()
        if g["is_opener"].any() else 0.0,
        "repair_per_100": 100 * g["repair"].mean(),
    }
    for who, mask in (("me", g["is_from_me"]), ("them", ~g["is_from_me"])):
        sub = g[mask]
        for h in HORSEMEN:
            out[f"{h}_{who}_per_100"] = 100 * sub[h].mean() if len(sub) else 0.0
    return out


LABELS = {
    "criticism": "Criticism", "contempt": "Contempt (⚠ strongest divorce predictor)",
    "defensiveness": "Defensiveness", "stonewalling": "Stonewalling",
    "repair": "Repair attempts (protective ✅)",
    "positivity_ratio": "Positivity ratio (5:1 = healthy)",
    "harsh_startup_rate": "Harsh startup rate",
}
