#!/usr/bin/env python3
"""
demo_data.py — Synthetic, fully deidentified dataset for the public demo dashboard.

No real messages. Generates a handful of fake relationships (a partner, a friend,
a parent, a child) with varied tone so every section of the dashboard lights up.
Run the demo with:  MSGANALYTICS_DEMO=1 streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fake people: (display name, seed, n messages, start date, warmth bias).
PEOPLE = [
    ("Alex (partner)", 1, 1600, "2022-01-01", 0.62),
    ("Sam", 2, 700, "2022-06-01", 0.70),
    ("Mom", 3, 900, "2022-03-01", 0.55),
    ("Kid A", 4, 600, "2023-01-01", 0.66),
    ("Jordan", 5, 400, "2023-04-01", 0.50),
]
CHILDREN = ["Kid A"]
ME_NAME = "You"

POS = ["love you", "haha that's great", "thank you so much", "miss you ❤️",
       "so proud of you", "yes! sounds perfect", "can't wait 😍", "you're the best",
       "great job today", "i appreciate you", "you've got this", "congrats!! 🎉"]
NEG = ["you always do this", "i'm so frustrated", "whatever", "you forgot again",
       "that's not fair", "you never listen", "ugh terrible day", "you said you would",
       "i don't believe you", "why didn't you", "i'm sorry, my fault"]
NEUTRAL = ["on my way", "what time?", "ok", "i'll grab milk", "see you at 6",
           "running late", "where are you", "i'll handle it", "sounds good",
           "did you call them", "let's plan for saturday"]


def _thread(name, seed, n, start, warmth):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="11h")
    rows = []
    for i, dt in enumerate(dates):
        w = warmth + 0.2 * np.sin(i / 150) - (0.3 if 0.45 * n < i < 0.6 * n else 0)
        roll = rng.random()
        pool = POS if roll < w else NEG if roll > 0.82 else NEUTRAL
        text = str(rng.choice(pool))
        rows.append({
            "dt": dt, "contact": name, "chat_id": seed, "chat_label": name,
            "is_group": False, "is_from_me": bool(i % 2),
            "direction": "sent" if i % 2 else "received", "text": text,
            "service": "iMessage", "n_chars": len(text), "n_words": len(text.split()),
            "has_emoji": any(ord(c) > 10000 for c in text),
            "n_emoji": sum(ord(c) > 10000 for c in text),
        })
    return rows


def demo_dataframe() -> pd.DataFrame:
    rows = []
    for name, seed, n, start, warmth in PEOPLE:
        rows += _thread(name, seed, n, start, warmth)
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"])
    return df.sort_values("dt").reset_index(drop=True)
