#!/usr/bin/env python3
"""
signals.py — Compute the 13 relationship signals as time series.

Given the tidy message table from extract.py (filtered to one contact), this
produces a resampled DataFrame indexed by time period with one column per
signal, ready to plot with time on the x-axis.

The 13 signals:
     1. volume            messages per period
     2. net_sentiment     mean VADER compound score (-1..+1)
     3. pct_positive      share of messages with compound >= 0.05
     4. pct_negative      share of messages with compound <= -0.05
     5. emoji_rate        emojis per message
     6. avg_words         mean words per message
     7. reciprocity       your share of messages (0.5 == balanced)
     8. initiation_share  your share of conversation-openers
     9. reply_latency_min median minutes to reply (cross-party)
    10. question_rate     share of messages containing '?'
    11. affection         affection-lexicon hits per message
    12. late_night_share  share of messages sent 10pm–4am
    13. media_share        share of messages that are attachments/no-text

Sentiment uses VADER (MIT license) — commercially redistributable.
"""

from __future__ import annotations

import re

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_ANALYZER = SentimentIntensityAnalyzer()

# Gap (minutes) of silence that marks the start of a new "conversation".
CONVERSATION_GAP_MIN = 60

# Affection lexicon — explicit warmth markers. Lowercased, word-boundary matched.
_AFFECTION_WORDS = {
    "love", "loved", "loving", "miss", "missed", "missing", "babe", "baby",
    "honey", "sweetheart", "darling", "dear", "xoxo", "hug", "hugs", "kiss",
    "kisses", "adore", "cherish", "forever", "soulmate", "beautiful",
    "handsome", "cutie", "sweetie", "❤", "❤️", "😘", "🥰", "😍", "💕", "💗",
    "💖", "💞", "💓", "💘", "♥",
}
_WORD_RE = re.compile(r"[a-zA-Z']+")
_EMOJI_AFFECTION = {w for w in _AFFECTION_WORDS if not w.isascii()}

# Practical / logistics lexicon — coordination, errands, money, scheduling.
_PRACTICAL_WORDS = {
    "schedule", "appointment", "meeting", "pick", "drop", "store", "grocery",
    "groceries", "dinner", "lunch", "address", "pay", "paid", "bill", "rent",
    "money", "card", "bank", "car", "gas", "buy", "order", "deliver", "delivery",
    "time", "when", "where", "today", "tomorrow", "tonight", "morning", "pm",
    "am", "min", "minutes", "hour", "leaving", "leave", "home", "work", "call",
    "text", "send", "need", "get", "got", "bring", "ready", "done", "ok", "okay",
}

# Vulnerability lexicon — self-disclosure of feelings, fears, needs, apologies.
_VULNERABLE_PHRASES = (
    "i feel", "i felt", "i'm scared", "im scared", "i'm afraid", "im afraid",
    "i'm sorry", "im sorry", "i'm worried", "im worried", "i'm anxious",
    "im anxious", "i'm nervous", "i need you", "i need", "i miss you", "i miss",
    "honestly", "to be honest", "i'm hurt", "im hurt", "i'm sad", "im sad",
    "i'm struggling", "im struggling", "i trust you", "i was wrong",
    "my fault", "i'm vulnerable", "i'm overwhelmed", "im overwhelmed",
    "i love you", "i can't", "i cant", "i'm lonely", "im lonely",
)


def _practical_score(text: str) -> int:
    if not text:
        return 0
    words = set(_WORD_RE.findall(text.lower()))
    hits = len(words & _PRACTICAL_WORDS)
    if re.search(r"\d", text):  # times, amounts, addresses
        hits += 1
    return hits


def _vulnerability_score(text: str) -> int:
    if not text:
        return 0
    low = text.lower()
    return sum(low.count(p) for p in _VULNERABLE_PHRASES)


# Encouragement / compliments — praise and support directed at the other person.
_ENCOURAGEMENT = [
    r"proud of you", r"i'?m proud", r"good job", r"great job", r"well done",
    r"you'?re amazing", r"you'?re the best", r"you can do it", r"you'?ve got this",
    r"you got this", r"i believe in you", r"so smart", r"so talented",
    r"you'?re great", r"nailed it", r"impressive", r"you'?re awesome", r"way to go",
    r"congrats", r"congratulations", r"you'?re doing (great|amazing|so well|so good)",
    r"keep it up", r"you rock", r"so proud", r"you'?re brilliant", r"you'?re incredible",
    r"you'?re wonderful", r"\bbeautiful\b", r"\bhandsome\b", r"you'?re killing it",
    r"i'?m so happy for you", r"you deserve", r"that'?s awesome",
]
_ENC_RE = [re.compile(p) for p in _ENCOURAGEMENT]


def _encouragement_hits(text: str) -> int:
    if not text:
        return 0
    t = text.lower()
    return sum(1 for r in _ENC_RE if r.search(t))


def _affection_hits(text: str) -> int:
    if not text:
        return 0
    words = set(_WORD_RE.findall(text.lower()))
    hits = len(words & _AFFECTION_WORDS)
    hits += sum(text.count(e) for e in _EMOJI_AFFECTION)
    return hits


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-message derived columns needed for the signals."""
    df = df.copy()
    df["compound"] = df["text"].apply(
        lambda t: _ANALYZER.polarity_scores(t)["compound"] if t else 0.0
    )
    df["is_positive"] = df["compound"] >= 0.05
    df["is_negative"] = df["compound"] <= -0.05
    df["affection"] = df["text"].apply(_affection_hits)
    df["encouragement"] = df["text"].apply(_encouragement_hits)
    df["has_question"] = df["text"].str.contains(r"\?", regex=True, na=False)
    df["hour"] = df["dt"].dt.hour
    df["is_late_night"] = (df["hour"] >= 22) | (df["hour"] < 4)
    df["is_media"] = df["n_chars"] == 0  # attachments / reactions have no text
    # Practical vs emotional: a message is "emotional" if it carries strong
    # sentiment or affection; "practical" if it's logistics-heavy. A message
    # can be neither (e.g. "lol"). emotional_balance = emo_share − prac_share.
    df["practical_score"] = df["text"].apply(_practical_score)
    df["is_emotional"] = (df["compound"].abs() >= 0.5) | (df["affection"] > 0)
    df["is_practical"] = (df["practical_score"] >= 2) & ~df["is_emotional"]
    df["vulnerability"] = df["text"].apply(_vulnerability_score)
    return df


def _reply_latencies(df: pd.DataFrame) -> pd.Series:
    """Minutes between a message and the next message from the *other* party.

    Indexed by the timestamp of the reply, so it can be resampled over time.
    """
    df = df.sort_values("dt")
    latencies = {}
    prev_dt = None
    prev_from_me = None
    for dt, from_me in zip(df["dt"], df["is_from_me"]):
        if prev_dt is not None and from_me != prev_from_me:
            mins = (dt - prev_dt).total_seconds() / 60.0
            # Ignore implausibly long gaps (different conversations entirely).
            if mins <= 24 * 60:
                latencies[dt] = mins
        prev_dt, prev_from_me = dt, from_me
    return pd.Series(latencies, name="reply_latency_min")


def _initiations(df: pd.DataFrame) -> pd.DataFrame:
    """Mark conversation-opening messages (first after a long silence)."""
    df = df.sort_values("dt").copy()
    gap = df["dt"].diff() > pd.Timedelta(minutes=CONVERSATION_GAP_MIN)
    df["is_opener"] = gap.fillna(True)  # very first message counts as an opener
    df["opener_by_me"] = df["is_opener"] & df["is_from_me"]
    return df


def compute_signals(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Return a time-indexed DataFrame with all 13 signals.

    freq: pandas offset alias — 'D' daily, 'W' weekly (default), 'ME' monthly.
    """
    if df.empty:
        return pd.DataFrame()

    df = enrich(df)
    df = _initiations(df)
    g = df.set_index("dt").groupby(pd.Grouper(freq=freq))

    out = pd.DataFrame(
        {
            "volume": g.size(),
            "net_sentiment": g["compound"].mean(),
            "pct_positive": g["is_positive"].mean(),
            "pct_negative": g["is_negative"].mean(),
            "emoji_rate": g["n_emoji"].mean(),
            "avg_words": g["n_words"].mean(),
            "reciprocity": g["is_from_me"].mean(),
            "question_rate": g["has_question"].mean(),
            "affection": g["affection"].mean(),
            "encouragement": g["encouragement"].mean(),
            "late_night_share": g["is_late_night"].mean(),
            "media_share": g["is_media"].mean(),
            # Extended signals (beyond the core 13):
            "emotional_balance": g["is_emotional"].mean() - g["is_practical"].mean(),
            "emotional_share": g["is_emotional"].mean(),
            "practical_share": g["is_practical"].mean(),
            "vulnerability": g["vulnerability"].mean(),
        }
    )

    # Positivity ratio (Gottman-inspired): positive ÷ negative message counts.
    # Gottman & Levenson (1992) found stable couples sustain ~5:1 positive-to-
    # negative interactions during conflict. This is a TEXT PROXY, not the lab
    # observational measure — framed as directional, not diagnostic.
    pos = g["is_positive"].sum()
    neg = g["is_negative"].sum()
    out["positivity_ratio"] = (pos / neg.replace(0, pd.NA)).clip(upper=20)

    # Initiation share: of conversation openers in each period, what share are mine.
    openers = df[df["is_opener"]].set_index("dt").groupby(pd.Grouper(freq=freq))
    out["initiation_share"] = openers["opener_by_me"].mean()

    # Reply latency: median per period.
    lat = _reply_latencies(df)
    if not lat.empty:
        out["reply_latency_min"] = lat.groupby(pd.Grouper(freq=freq)).median()
    else:
        out["reply_latency_min"] = pd.NA

    # Keep only periods that actually had messages.
    out = out[out["volume"] > 0]
    return out


# Signals that are meaningful *per person* (split into You vs Them lines).
PER_PERSON_SIGNALS = [
    "volume", "net_sentiment", "pct_positive", "pct_negative", "emoji_rate",
    "avg_words", "question_rate", "affection", "encouragement", "late_night_share",
    "media_share", "vulnerability", "reply_latency_min",
]
# Signals that are inherently relational (a single line for the pair).
RELATIONAL_SIGNALS = [
    "reciprocity", "initiation_share", "positivity_ratio", "emotional_balance",
]


def compute_signals_split(df: pd.DataFrame, freq: str = "W") -> dict:
    """Per-person time series for the PER_PERSON_SIGNALS.

    Returns {"me": DataFrame, "them": DataFrame}, each indexed by period with
    one column per per-person signal. Lets the UI plot You vs Them in two
    colors instead of averaging both together.
    """
    if df.empty:
        return {"me": pd.DataFrame(), "them": pd.DataFrame()}

    df = enrich(df).sort_values("dt")
    # Per-person reply latency: a "reply" is a message whose sender differs
    # from the previous one; the latency belongs to whoever sent the reply.
    df = df.reset_index(drop=True)
    prev_from_me = df["is_from_me"].shift()
    gap_min = df["dt"].diff().dt.total_seconds() / 60.0
    is_reply = (df["is_from_me"] != prev_from_me) & prev_from_me.notna() & (gap_min <= 24 * 60)
    df["reply_latency_min"] = gap_min.where(is_reply)

    out = {}
    for who, mask in (("me", df["is_from_me"]), ("them", ~df["is_from_me"])):
        sub = df[mask].set_index("dt")
        g = sub.groupby(pd.Grouper(freq=freq))
        d = pd.DataFrame(
            {
                "volume": g.size(),
                "net_sentiment": g["compound"].mean(),
                "pct_positive": g["is_positive"].mean(),
                "pct_negative": g["is_negative"].mean(),
                "emoji_rate": g["n_emoji"].mean(),
                "avg_words": g["n_words"].mean(),
                "question_rate": g["has_question"].mean(),
                "affection": g["affection"].mean(),
                "encouragement": g["encouragement"].mean(),
                "late_night_share": g["is_late_night"].mean(),
                "media_share": g["is_media"].mean(),
                "vulnerability": g["vulnerability"].mean(),
                "reply_latency_min": g["reply_latency_min"].median(),
            }
        )
        out[who] = d[d["volume"] > 0]
    return out


# Friendly metadata for the UI: label, description, and whether higher is "warmer".
SIGNAL_META = {
    "volume": ("Message volume", "Messages per period — overall activity."),
    "net_sentiment": ("Net sentiment", "Mean VADER compound score (−1 to +1)."),
    "pct_positive": ("% positive", "Share of clearly-positive messages."),
    "pct_negative": ("% negative", "Share of clearly-negative messages."),
    "emoji_rate": ("Emoji rate", "Emojis per message — playfulness."),
    "avg_words": ("Avg words / msg", "Verbosity / investment."),
    "reciprocity": ("Your share", "Fraction of messages you sent (0.5 = balanced)."),
    "initiation_share": ("You initiate", "Share of conversations you opened."),
    "reply_latency_min": ("Reply latency (min)", "Median minutes to reply."),
    "question_rate": ("Question rate", "Share of messages asking something."),
    "affection": ("Affection", "Affection-word/emoji hits per message."),
    "encouragement": ("Encouragement / compliments",
                      "Praise & support per message — how complimentary each of you is."),
    "late_night_share": ("Late-night share", "Share sent 10pm–4am."),
    "media_share": ("Media share", "Share that are photos/links/attachments."),
    # Extended pair (beyond the core 13):
    "emotional_balance": (
        "Emotional ↔ practical",
        "Emotional share minus practical share. >0 = more emotional/feelings, "
        "<0 = more logistics/coordination.",
    ),
    "vulnerability": (
        "Vulnerability",
        "Self-disclosure of feelings, fears, needs & apologies per message "
        "(\"I feel…\", \"I'm sorry\", \"I need you\"…).",
    ),
    "positivity_ratio": (
        "Positivity ratio (Gottman-inspired)",
        "Positive ÷ negative messages. Gottman's research finds healthy couples "
        "sustain ~5:1. Text proxy — directional, not diagnostic.",
    ),
}

# The core 13 — kept intact and ordered. The extended pair is tracked separately
# so the significant count of 13 stays meaningful in the UI.
CORE_SIGNALS = [
    "volume", "net_sentiment", "pct_positive", "pct_negative", "emoji_rate",
    "avg_words", "reciprocity", "initiation_share", "reply_latency_min",
    "question_rate", "affection", "late_night_share", "media_share",
]
EXTENDED_SIGNALS = ["emotional_balance", "vulnerability", "positivity_ratio"]
ORDERED_SIGNALS = CORE_SIGNALS + EXTENDED_SIGNALS
assert len(CORE_SIGNALS) == 13, "Core signal count must stay at 13."
assert all(k in SIGNAL_META for k in ORDERED_SIGNALS)
