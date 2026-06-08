#!/usr/bin/env python3
"""
personality.py — Text-based MBTI-style type indicator.

Estimates the four MBTI dichotomies from a person's writing style and reports
percentages plus a 4-letter type, in the same format as a standard MBTI readout.
Note: this infers tendencies from *language use* rather than a formal
questionnaire, so it's best read as a complementary indicator alongside a proper
assessment — but it draws on the same four dichotomies a clinician would use.

Dichotomies & proxies (each side a small lexicon; percentage = side/(both)):
  E/I  — Extraversion: social/“we” words, exclamations, emoji, initiation.
  N/S  — Intuition: abstract/idea words  vs  concrete/sensory/detail words.
  F/T  — Feeling: emotion/value words     vs  logic/reason words.
  P/J  — Perceiving: spontaneity words     vs  planning/structure words.
"""

from __future__ import annotations

import re

import pandas as pd

import signals as sig

LEX = {
    "E": [r"\bwe\b", r"\bus\b", r"everyone", r"\bparty\b", r"\bfun\b", r"hang out",
          r"\btogether\b", r"y'?all", r"\bguys\b", r"let'?s", r"\bpeople\b", r"so excited"],
    "I": [r"\balone\b", r"\bquiet\b", r"\btired\b", r"by myself", r"\bmyself\b",
          r"need space", r"stay (in|home)", r"rather not", r"\bexhausted\b", r"recharge"],
    "N": [r"\bidea\b", r"imagine", r"\bmaybe\b", r"possibilit", r"\bfuture\b", r"\bdream\b",
          r"\bconcept\b", r"\btheory\b", r"meaning", r"\bwhy\b", r"\bwonder\b", r"what if",
          r"big picture", r"\bcould be\b"],
    "S": [r"\bsaw\b", r"\bexact\b", r"specific", r"\bactual", r"\bdetail", r"\bnow\b",
          r"\breal\b", r"\bfact", r"literally", r"\btoday\b", r"practical", r"step by step",
          r"\bexactly\b"],
    "F": [r"\bfeel", r"\blove\b", r"\bcare\b", r"\bhurt\b", r"\bhappy\b", r"\bsad\b",
          r"\bsorry\b", r"\bheart\b", r"\bmiss\b", r"emotional", r"\bupset\b", r"\bexcited\b",
          r"i'?m here for"],
    "T": [r"\bthink\b", r"because", r"\blogic", r"\breason", r"analy", r"makes sense",
          r"efficient", r"\bobjective\b", r"\brational", r"figure out", r"\bfair\b",
          r"the point is"],
    "P": [r"\bmaybe\b", r"whatever", r"\blater\b", r"flexible", r"spontaneous", r"we'?ll see",
          r"\bidk\b", r"whenever", r"up in the air", r"wing it", r"last minute", r"\bso\b.*\?"],
    "J": [r"\bplan\b", r"schedule", r"\blist\b", r"deadline", r"\bshould\b", r"\bmust\b",
          r"organi", r"\bready\b", r"\bfinish", r"on time", r"\bdecide", r"settled", r"by then"],
}
_COMP = {k: [re.compile(p) for p in v] for k, v in LEX.items()}


def _hits(patterns, text):
    return sum(1 for p in patterns if p.search(text))


# Each axis is one continuous "leaning" signal = rate of that pole's markers per
# message. Higher than your population average → that letter; lower → its opposite.
#   POLE_E (vs I), POLE_N (vs S), POLE_F (vs T), POLE_J (vs P)
_POLE = {"E": "E", "N": "N", "F": "F", "J": "J"}


def _raw_signals(df_person: pd.DataFrame) -> dict:
    """Per-message rate of each pole's markers for one person."""
    n = max(len(df_person), 1)
    low = df_person["text"].fillna("").str.lower()
    sig_vals = {}
    for ax in ("E", "N", "F", "J"):
        hits = int(low.apply(lambda t, c=_COMP[ax]: _hits(c, t)).sum())
        sig_vals[ax] = hits / n
    # Extraversion also picks up exclamations + emoji.
    sig_vals["E"] += (df_person["text"].str.count("!").sum()
                      + df_person["n_emoji"].sum()) / n
    return sig_vals


def population(df_all: pd.DataFrame, min_msgs: int = 40) -> dict:
    """Mean & std of each axis signal across all 1:1 contacts — the calibration baseline."""
    d = sig.enrich(df_all[~df_all["is_group"]].copy())
    per_axis = {ax: [] for ax in ("E", "N", "F", "J")}
    for _, g in d.groupby("contact"):
        if len(g) < min_msgs:
            continue
        s = _raw_signals(g)
        for ax in per_axis:
            per_axis[ax].append(s[ax])
    base = {}
    for ax, vals in per_axis.items():
        if len(vals) >= 2:
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / len(vals)
            base[ax] = (m, var ** 0.5 or 1e-9)
        else:
            base[ax] = (0.0, 1e-9)
    return base


def _calibrated_pct(x: float, base_ax) -> float:
    """Convert a raw signal to a 0–100% leaning relative to the population (z-score)."""
    m, sd = base_ax
    z = (x - m) / sd if sd else 0.0
    return round(max(2.0, min(98.0, 50 + 20 * z)), 1)


def _person(df_person: pd.DataFrame, baseline: dict) -> dict:
    if df_person.empty:
        return {}
    s = _raw_signals(df_person)
    eP = _calibrated_pct(s["E"], baseline["E"])
    nP = _calibrated_pct(s["N"], baseline["N"])
    fP = _calibrated_pct(s["F"], baseline["F"])
    jP = _calibrated_pct(s["J"], baseline["J"])
    typ = ("E" if eP >= 50 else "I") + ("N" if nP >= 50 else "S") \
        + ("F" if fP >= 50 else "T") + ("J" if jP >= 50 else "P")
    return {
        "E": eP, "I": round(100 - eP, 1), "N": nP, "S": round(100 - nP, 1),
        "F": fP, "T": round(100 - fP, 1), "J": jP, "P": round(100 - jP, 1),
        "type": typ, "n": len(df_person),
    }


def analyze(df: pd.DataFrame, baseline: dict | None = None) -> dict:
    """Return {'me': {...}, 'them': {...}}, calibrated against `baseline`.

    baseline comes from population() over all the user's contacts, so each letter
    means "more X than your typical conversation" — which makes types discriminate
    (otherwise everyone collapses to the same warm/social/practical type).
    """
    if df.empty:
        return {"me": {}, "them": {}}
    if baseline is None:
        baseline = population(df)   # fallback: calibrate within this thread only
    d = sig.enrich(df)
    return {"me": _person(d[d["is_from_me"]], baseline),
            "them": _person(d[~d["is_from_me"]], baseline)}


AXES = [("E", "I"), ("N", "S"), ("F", "T"), ("P", "J")]

FULL_NAME = {
    "E": "Extraverted", "I": "Introverted", "N": "Intuitive", "S": "Sensing",
    "F": "Feeling", "T": "Thinking", "P": "Perceiving", "J": "Judging",
}

TYPE_DESC = {
    "INTJ": "Architect — strategic, independent, long-range planner.",
    "INTP": "Logician — analytical, curious, idea-driven.",
    "ENTJ": "Commander — decisive, organizing, goal-oriented.",
    "ENTP": "Debater — inventive, quick, enjoys possibilities.",
    "INFJ": "Advocate — insightful, principled, quietly determined.",
    "INFP": "Mediator — caring, values-driven, idealistic.",
    "ENFJ": "Protagonist — warm, encouraging, people-focused.",
    "ENFP": "Campaigner — enthusiastic, imaginative, expressive.",
    "ISTJ": "Logistician — dependable, practical, detail-oriented.",
    "ISFJ": "Defender — loyal, supportive, conscientious.",
    "ESTJ": "Executive — organized, direct, takes charge.",
    "ESFJ": "Consul — sociable, caring, attentive to others.",
    "ISTP": "Virtuoso — pragmatic, hands-on, calm problem-solver.",
    "ISFP": "Adventurer — gentle, present-focused, aesthetic.",
    "ESTP": "Entrepreneur — energetic, spontaneous, action-oriented.",
    "ESFP": "Entertainer — lively, warm, in-the-moment.",
}
