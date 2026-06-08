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


def _person(df_person: pd.DataFrame) -> dict:
    if df_person.empty:
        return {}
    low = df_person["text"].fillna("").str.lower()
    counts = {k: int(low.apply(lambda t, c=_COMP[k]: _hits(c, t)).sum()) for k in LEX}
    # Extra E signal: exclamations + emoji.
    counts["E"] += int(df_person["text"].str.count("!").sum())
    counts["E"] += int(df_person["n_emoji"].sum())

    def pct(a, b):
        tot = counts[a] + counts[b]
        return 50.0 if tot == 0 else round(100 * counts[a] / tot, 1)

    eP, nP, fP, pP = pct("E", "I"), pct("N", "S"), pct("F", "T"), pct("P", "J")
    typ = ("E" if eP >= 50 else "I") + ("N" if nP >= 50 else "S") \
        + ("F" if fP >= 50 else "T") + ("P" if pP >= 50 else "J")
    return {
        "E": eP, "I": round(100 - eP, 1), "N": nP, "S": round(100 - nP, 1),
        "F": fP, "T": round(100 - fP, 1), "P": pP, "J": round(100 - pP, 1),
        "type": typ, "n": len(df_person),
    }


def analyze(df: pd.DataFrame) -> dict:
    """Return {'me': {...}, 'them': {...}} personality estimates for a 1:1 thread."""
    if df.empty:
        return {"me": {}, "them": {}}
    d = sig.enrich(df)
    return {"me": _person(d[d["is_from_me"]]), "them": _person(d[~d["is_from_me"]])}


AXES = [("E", "I"), ("N", "S"), ("F", "T"), ("P", "J")]

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
