#!/usr/bin/env python3
"""
generate_samples.py — Render DEIDENTIFIED sample charts for the public README.

Uses 100% synthetic message data (fake names "You" / "Alex", invented text) run
through the real analysis modules, so the screenshots show exactly what the tool
produces WITHOUT exposing any real messages. Output: docs/*.png (safe to commit).

    python3 generate_samples.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import signals as sig
import dynamics as dyn
import gottman

OUT = "docs"
os.makedirs(OUT, exist_ok=True)

DARK = dict(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="#c8c8d0"), margin=dict(l=40, r=20, t=50, b=40),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"))
ME, THEM = "#3aa0ff", "#ff5fa2"

POS = ["love you", "haha that's great", "thank you so much", "miss you ❤️",
       "so proud of you", "yes! sounds perfect", "can't wait to see you 😍",
       "you're the best", "great job today"]
NEG = ["you always do this", "i'm so frustrated", "whatever", "you forgot again",
       "that's not fair", "i'm sorry, my fault", "you never listen", "ugh terrible day"]
NEUTRAL = ["on my way", "what time?", "ok", "i'll grab milk", "see you at 6",
           "did you call the plumber", "running late", "where are you"]


def synth(seed: int, n: int = 1200, start="2024-01-01"):
    """Build a synthetic 1:1 message table matching extract.py's schema."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="9h")
    rows = []
    for i, dt in enumerate(dates):
        # Slow drift: warmth rises over time; a rough patch mid-series.
        warmth = 0.6 + 0.25 * np.sin(i / 180) - (0.3 if 400 < i < 520 else 0)
        roll = rng.random()
        pool = POS if roll < warmth else NEG if roll > 0.85 else NEUTRAL
        text = str(rng.choice(pool))
        rows.append({
            "dt": dt, "contact": "Alex", "chat_id": 1, "chat_label": "Alex",
            "is_group": False, "is_from_me": bool(i % 2),
            "direction": "sent" if i % 2 else "received", "text": text,
            "service": "iMessage", "n_chars": len(text), "n_words": len(text.split()),
            "has_emoji": any(ord(c) > 10000 for c in text),
            "n_emoji": sum(ord(c) > 10000 for c in text),
        })
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"])
    return df


def save(fig, name):
    fig.update_layout(**DARK, showlegend=True,
                      legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"))
    fig.write_image(os.path.join(OUT, name), width=900, height=460, scale=2)
    print("wrote", os.path.join(OUT, name))


def fit_line(idx, vals, color, name):
    s = pd.Series(vals).dropna()
    x = idx[s.index]
    t = go.Scatter(x=x, y=s.values, mode="markers", name=name,
                   marker=dict(size=7, color=color, opacity=0.9,
                               line=dict(width=1, color="rgba(255,255,255,0.8)")))
    xn = np.array([pd.Timestamp(v).toordinal() for v in x])
    m, b = np.polyfit(xn, s.values, 1)
    xl = np.array([xn.min(), xn.max()])
    line = go.Scatter(x=[pd.Timestamp.fromordinal(int(v)) for v in xl], y=m * xl + b,
                      mode="lines", line=dict(width=2.5, color=color), showlegend=False)
    return t, line


def main():
    df = synth(7)

    # 1) Net sentiment over time, split by person.
    sp = sig.compute_signals_split(df, "W")
    fig = go.Figure()
    for t in fit_line(sp["me"].index, sp["me"]["net_sentiment"].values, ME, "You"):
        fig.add_trace(t)
    for t in fit_line(sp["them"].index, sp["them"]["net_sentiment"].values, THEM, "Alex"):
        fig.add_trace(t)
    fig.update_layout(title="Net sentiment over time (per person)")
    save(fig, "sample_sentiment.png")

    # 2) Push–pull index.
    pp = dyn.compute_push_pull(df, "W")
    pol = pp["polarization"]
    sm = pol.rolling(6, center=True, min_periods=1).mean()
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=pol.max() * 1.2 + 0.1, fillcolor="rgba(58,160,255,0.07)", line_width=0)
    fig.add_hrect(y0=pol.min() * 1.2 - 0.1, y1=0, fillcolor="rgba(255,95,162,0.07)", line_width=0)
    fig.add_trace(go.Scatter(x=pol.index, y=pol.values, mode="markers",
                             marker=dict(size=5, color="#888"), name="per week"))
    fig.add_trace(go.Scatter(x=sm.index, y=sm.values, mode="lines",
                             line=dict(width=4, color="#9467bd"), name="trend"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="Push–pull index (▲ you pursue · ▼ they pursue)")
    save(fig, "sample_pushpull.png")

    # 3) Gottman Four Horsemen — smoothed rate per 100 msgs.
    g = gottman.compute_gottman(df, "ME")
    fig = go.Figure()
    for who, color, nm in (("me", ME, "You"), ("them", THEM, "Alex")):
        gd = g[who]
        w = max(3, len(gd) // 6)
        horse = gd[gottman.HORSEMEN].sum(axis=1)
        rate = 100 * horse.rolling(w, min_periods=1).sum() / gd["volume"].rolling(w, min_periods=1).sum()
        fig.add_trace(go.Scatter(x=rate.index, y=rate.values, mode="lines",
                                 line=dict(width=3, color=color), name=nm))
    fig.update_layout(title="Conflict signals (Four Horsemen) — rate per 100 messages")
    save(fig, "sample_gottman.png")

    # 4) Who-leads contagion bars.
    wl = dyn.who_leads(df, "ME")
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Negativity", marker_color="#E74C3C",
                         x=["Alex mirrors You", "You mirror Alex"],
                         y=[100 * wl["neg_me_to_them"], 100 * wl["neg_them_to_me"]]))
    fig.add_trace(go.Bar(name="Positivity", marker_color="#2ca02c",
                         x=["Alex mirrors You", "You mirror Alex"],
                         y=[100 * wl["pos_me_to_them"], 100 * wl["pos_them_to_me"]]))
    fig.update_layout(title="Who leads the mood? (emotional contagion)", barmode="group")
    save(fig, "sample_wholeads.png")

    print("\nAll sample charts written to docs/ (synthetic data — safe to publish).")


if __name__ == "__main__":
    main()
