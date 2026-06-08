"""
aliases.py — Merge multiple identifiers into one canonical person.

People text from several handles (a phone number AND an email, two numbers,
etc.), and the address book may even store them as separate contact cards.
aliases.json maps a canonical display name to its list of identifiers:

    { "Alex": ["alex@example.com", "5551234567"] }

We remap the `contact` column so every thread for the same person collapses
into one, and the dashboard treats them as a single relationship.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from contacts import _norm_phone

ALIASES_FILE = os.path.join(os.path.dirname(__file__), "aliases.json")


def load() -> dict:
    if os.path.exists(ALIASES_FILE):
        try:
            with open(ALIASES_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def reverse_map() -> dict:
    """normalized identifier -> canonical name (ignores reserved _keys)."""
    rev = {}
    for canon, ids in load().items():
        if canon.startswith("_"):
            continue
        for i in ids:
            rev[_norm_phone(i)] = canon
    return rev


def excluded() -> set:
    """Normalized identifiers to drop from analysis entirely."""
    return {_norm_phone(i) for i in load().get("_exclude", [])}


def excluded_prefixes() -> list:
    """Phone area-code prefixes to drop (e.g. toll-free 800/844/etc.)."""
    return [str(p) for p in load().get("_exclude_prefixes", [])]


def apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Drop excluded identifiers, then remap 1:1 contacts to canonical names."""
    if df.empty:
        return df
    df = df.copy()
    norm = df["contact"].astype(str).map(_norm_phone)
    # Exact-identifier exclusions (1:1 only).
    excl = excluded()
    drop = norm.isin(excl) & (~df["is_group"]) if excl else pd.Series(False, index=df.index)
    # Prefix exclusions: 10-digit phone numbers whose area code matches (toll-free, etc.).
    prefixes = excluded_prefixes()
    if prefixes:
        is_phone = ~norm.str.contains("@", na=False) & (norm.str.len() == 10)
        pref_drop = is_phone & norm.str.slice(0, 3).isin(prefixes) & (~df["is_group"])
        drop = drop | pref_drop
    if drop.any():
        df = df[~drop].copy()
        norm = norm[~drop]
    # Aliases.
    rev = reverse_map()
    if rev:
        mapped = norm.map(rev)
        use = mapped.notna() & (~df["is_group"])
        df.loc[use, "contact"] = mapped[use]
    return df


def is_alias(name: str) -> bool:
    return name in load() and not name.startswith("_")


def primary_identifier(canonical: str) -> str:
    """A representative identifier (for photo/name lookup) for a canonical name."""
    ids = load().get(canonical)
    return ids[0] if ids else canonical
