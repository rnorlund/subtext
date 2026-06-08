#!/usr/bin/env python3
"""
extract.py — Load ALL messages across every contact into one tidy table.

Produces a pandas DataFrame (and caches it to messages.parquet) with one row
per message and the columns the signal engine needs:

    dt            timezone-aware datetime (local)
    contact       the other party's identifier (phone/email); '' for group/me-only
    chat_id       which chat it belongs to
    chat_label    human-friendly chat name
    is_from_me    bool
    direction     'sent' | 'received'
    text          decoded message text ('' if none)
    service       iMessage / SMS
    n_chars       len(text)
    n_words       word count
    has_emoji     bool
    n_emoji       emoji count

Run directly to (re)build the cache and print a coverage summary:
    python3 extract.py
"""

from __future__ import annotations

import re
import sys

import pandas as pd

import messages_lib as ml

CACHE = "messages.parquet"

# Emoji detection: broad unicode ranges covering the common emoji blocks.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols, pictographs, supplemental, extended-A
    "\U00002600-\U000027BF"  # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flags)
    "\U00002190-\U000021FF"  # arrows
    "\U00002B00-\U00002BFF"  # misc symbols and arrows
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0001F000-\U0001F0FF"  # mahjong / dominoes / cards
    "]"
)


def count_emoji(text: str) -> int:
    return len(_EMOJI_RE.findall(text)) if text else 0


def build_dataframe(db_path: str = ml.DEFAULT_DB) -> pd.DataFrame:
    """Snapshot the DB and load every message into a DataFrame."""
    snap = ml.snapshot_db(db_path)
    conn = ml.connect(snap)

    chats = {c.chat_id: c for c in ml.list_chats(conn)}
    rows = []
    for chat_id, chat in chats.items():
        # For 1:1 chats the contact is the single member; for groups, blank.
        contact = chat.identifiers[0] if len(chat.identifiers) == 1 else ""
        for msg in ml.iter_messages(conn, chat_id):
            text = msg["text"] or ""
            rows.append(
                {
                    "dt": msg["dt"],
                    "contact": contact or chat.label,
                    "chat_id": chat_id,
                    "chat_label": chat.label,
                    "is_group": len(chat.identifiers) > 1,
                    "is_from_me": msg["is_from_me"],
                    "direction": "sent" if msg["is_from_me"] else "received",
                    "text": text,
                    "service": msg["service"] or "",
                    "n_chars": len(text),
                    "n_words": len(text.split()),
                    "has_emoji": count_emoji(text) > 0,
                    "n_emoji": count_emoji(text),
                }
            )
    # NOTE: we intentionally do NOT recover "orphaned" messages (those without a
    # chat_message_join row). After an iCloud re-sync many old messages lose their
    # chat link, and recovering them by sender handle yields ONE-SIDED history
    # (received-only — sent messages have no handle to re-attribute), which distorts
    # every two-sided metric. We keep only properly-paired, chat-joined messages.
    conn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["dt"] = pd.to_datetime(df["dt"], utc=True).dt.tz_convert(None)
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    return df


def load_cached(rebuild: bool = False, db_path: str = ml.DEFAULT_DB) -> pd.DataFrame:
    """Load the parquet cache, building it first if missing or rebuild=True."""
    import os

    if not rebuild and os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    df = build_dataframe(db_path)
    if not df.empty:
        df.to_parquet(CACHE, index=False)
    return df


def coverage_summary(df: pd.DataFrame) -> str:
    """Human-readable summary to gut-check completeness vs the iPhone."""
    if df.empty:
        return "No messages found."
    total = len(df)
    earliest = df["dt"].min()
    latest = df["dt"].max()
    imsg = (df["service"].str.lower() == "imessage").sum()
    sms = (df["service"].str.lower() == "sms").sum()
    n_contacts = df.loc[~df["is_group"], "contact"].nunique()
    text_msgs = (df["n_chars"] > 0).sum()
    lines = [
        f"Total messages:      {total:,}",
        f"With decodable text: {text_msgs:,}  ({100*text_msgs/total:.0f}%)",
        f"Date range:          {earliest:%Y-%m-%d}  →  {latest:%Y-%m-%d}",
        f"iMessage / SMS:      {imsg:,} / {sms:,}",
        f"Distinct 1:1 contacts: {n_contacts:,}",
        "",
        "iPhone-gap check:",
        f"  • SMS share is {100*sms/total:.0f}%. If that looks low, green-bubble texts",
        "    may live only on your iPhone (Text Message Forwarding not enabled).",
        f"  • Earliest message is {earliest:%Y-%m-%d}. If you've texted longer than that,",
        "    the Mac was set up later and older history may be iPhone/iCloud-only.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        df = build_dataframe()
    except PermissionError:
        print(
            "❌ Operation not permitted. Grant Full Disk Access to VS Code / your\n"
            "   terminal (System Settings → Privacy & Security → Full Disk Access),\n"
            "   then fully quit (Cmd+Q) and reopen.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if df.empty:
        print("No messages found in the database.")
        raise SystemExit(0)
    df.to_parquet(CACHE, index=False)
    print(f"Cached {len(df):,} messages → {CACHE}\n")
    print(coverage_summary(df))
