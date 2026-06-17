#!/usr/bin/env python3
"""
wrapped.py — A shareable "Relationship Wrapped" summary card.

Composes headline insights for one relationship into a single square image
(no raw messages — just the pretty stats), à la Spotify Wrapped. Meant to be
downloaded and shared, which is the app's organic growth loop.
"""

from __future__ import annotations

import collections

import pandas as pd
import plotly.graph_objects as go

import signals as sig
import dynamics as dyn
from extract import _EMOJI_RE


def summarize(sub: pd.DataFrame, me: str, them: str) -> dict:
    """Headline stats for the Wrapped card."""
    if sub.empty:
        return {}
    e = sig.enrich(sub)
    n = len(e)
    span_days = max((e["dt"].max() - e["dt"].min()).days, 1)
    pos, neg = int(e["is_positive"].sum()), int(e["is_negative"].sum())
    ratio = pos / neg if neg else float(pos or 0)
    by_month = e.set_index("dt").resample("ME").size()
    busiest = by_month.idxmax()
    # Top emoji across the thread.
    # Skip non-display codepoints: variation selectors, ZWJ, skin-tone modifiers.
    _skip = set(range(0xFE00, 0xFE10)) | {0x200D} | set(range(0x1F3FB, 0x1F400))
    emojis = collections.Counter()
    for t in e["text"].dropna():
        for ch in _EMOJI_RE.findall(t):
            if ord(ch) not in _skip:
                emojis[ch] += 1
    top_emoji = emojis.most_common(1)[0][0] if emojis else "💬"
    # Who pursues (composite balance, >0.5 = me).
    pb = dyn.compute_pursuit(sub, freq="ME")
    pursuit = pb["pursuit_balance"].mean() if not pb.empty else 0.5
    return {
        "messages": n,
        "per_day": round(n / span_days, 1),
        "span_days": span_days,
        "my_share": e["is_from_me"].mean(),
        "sentiment": e["compound"].mean(),
        "positivity_ratio": ratio,
        "emoji_total": int(e["n_emoji"].sum()),
        "top_emoji": top_emoji,
        "affection": int(e["affection"].sum()),
        "encouragement": int(e["encouragement"].sum()),
        "busiest_month": busiest.strftime("%B %Y"),
        "pursuer": me if pursuit > 0.53 else them if pursuit < 0.47 else "Both equally",
        "first": e["dt"].min().strftime("%b %Y"),
        "last": e["dt"].max().strftime("%b %Y"),
    }


def card_png(stats: dict, me: str, them: str) -> bytes:
    """Render the Wrapped card as a shareable square PNG."""
    rows = [
        ("💬 messages", f"{stats['messages']:,}"),
        ("📅 since", stats["first"]),
        ("⚖️ your share", f"{100*stats['my_share']:.0f}% {me} / {100*(1-stats['my_share']):.0f}% {them}"),
        ("😊 avg sentiment", f"{stats['sentiment']:+.2f}"),
        ("💚 positivity ratio", f"{stats['positivity_ratio']:.1f} : 1"),
        (f"{stats['top_emoji']} top emoji", f"{stats['emoji_total']:,} emojis sent"),
        ("❤️ affection notes", f"{stats['affection']:,}"),
        ("🔥 busiest month", stats["busiest_month"]),
        ("🧲 reaches out more", stats["pursuer"]),
    ]
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.95, xref="paper", yref="paper", showarrow=False,
                       text=f"<b>{them} &amp; {me}</b>", font=dict(size=40, color="#ffffff"))
    fig.add_annotation(x=0.5, y=0.89, xref="paper", yref="paper", showarrow=False,
                       text="Relationship Wrapped", font=dict(size=20, color="#9b8afb"))
    y = 0.80
    for label, val in rows:
        fig.add_annotation(x=0.08, y=y, xref="paper", yref="paper", showarrow=False,
                           text=label, font=dict(size=22, color="#b8b8c0"), xanchor="left")
        fig.add_annotation(x=0.92, y=y, xref="paper", yref="paper", showarrow=False,
                           text=f"<b>{val}</b>", font=dict(size=22, color="#ffffff"), xanchor="right")
        y -= 0.083
    fig.add_annotation(x=0.5, y=0.03, xref="paper", yref="paper", showarrow=False,
                       text="made with imessage-relationship-analytics · 100% on-device",
                       font=dict(size=14, color="#666"))
    fig.update_layout(
        width=900, height=900, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig.to_image(format="png", scale=2)
