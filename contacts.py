"""
contacts.py — Resolve phone/email identifiers to real names (and photos) from
the macOS Contacts (AddressBook) database.

Messages only stores raw handles (+15551234567, foo@bar.com). Humans think in
names and faces. This reads the local AddressBook SQLite stores (requires Full
Disk Access, which we already have) and builds a lookup:

    identifier (normalized)  ->  display name
    identifier (normalized)  ->  photo bytes (for the contacts that have one)

Photos live as JPEG/PNG blobs in ZABCDRECORD.ZIMAGEDATA / ZTHUMBNAILIMAGEDATA.
"""

from __future__ import annotations

import glob
import os
import shutil
import sqlite3
import tempfile

AB_GLOBS = [
    os.path.expanduser("~/Library/Application Support/AddressBook/Sources/*/AddressBook-v22.abcddb"),
    os.path.expanduser("~/Library/Application Support/AddressBook/AddressBook-v22.abcddb"),
]


def _norm_phone(s: str) -> str:
    """Last-10-digits key for a phone, or lowercased email."""
    s = (s or "").strip().lower()
    if "@" in s:
        return s
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits[-10:] if len(digits) >= 10 else digits


def _full_name(first, last, org, nick) -> str | None:
    name = " ".join(p for p in (first, last) if p).strip()
    return name or nick or org or None


def _load_one(db_path: str, names: dict, photos: dict) -> None:
    tmp = tempfile.mktemp(suffix=".abcddb")
    try:
        shutil.copy2(db_path, tmp)
        conn = sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        # Record id -> (name, photo)
        recs = {}
        for r in conn.execute(
            "SELECT Z_PK, ZFIRSTNAME, ZLASTNAME, ZORGANIZATION, ZNICKNAME, "
            "ZIMAGEDATA, ZTHUMBNAILIMAGEDATA FROM ZABCDRECORD"
        ):
            nm = _full_name(r["ZFIRSTNAME"], r["ZLASTNAME"], r["ZORGANIZATION"], r["ZNICKNAME"])
            photo = r["ZIMAGEDATA"] or r["ZTHUMBNAILIMAGEDATA"]
            recs[r["Z_PK"]] = (nm, photo)

        def attach(identifier, owner):
            rec = recs.get(owner)
            if not rec:
                return
            nm, photo = rec
            key = _norm_phone(identifier)
            if not key:
                return
            if nm and key not in names:
                names[key] = nm
            if photo and key not in photos:
                photos[key] = bytes(photo)

        for r in conn.execute("SELECT ZFULLNUMBER, ZOWNER FROM ZABCDPHONENUMBER"):
            if r["ZFULLNUMBER"]:
                attach(r["ZFULLNUMBER"], r["ZOWNER"])
        for r in conn.execute("SELECT ZADDRESS, ZOWNER FROM ZABCDEMAILADDRESS"):
            if r["ZADDRESS"]:
                attach(r["ZADDRESS"], r["ZOWNER"])
        conn.close()
    except Exception:
        pass  # skip unreadable/locked sources; partial data is fine
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


_NAMES: dict | None = None
_PHOTOS: dict | None = None


def _ensure_loaded() -> None:
    global _NAMES, _PHOTOS
    if _NAMES is not None:
        return
    _NAMES, _PHOTOS = {}, {}
    seen = set()
    for pattern in AB_GLOBS:
        for db in glob.glob(pattern):
            if db in seen:
                continue
            seen.add(db)
            _load_one(db, _NAMES, _PHOTOS)


def resolve_name(identifier: str) -> str | None:
    """Return the contact's display name for a phone/email, or None."""
    _ensure_loaded()
    return _NAMES.get(_norm_phone(identifier))


def get_photo(identifier: str) -> bytes | None:
    """Return clean JPEG/PNG bytes for a contact, or None.

    AddressBook prefixes the blob with a 1-byte type marker, so the real image
    starts a few bytes in. We seek the JPEG/PNG magic and slice from there;
    blobs that aren't raw images (e.g. reference types) return None.
    """
    _ensure_loaded()
    raw = _PHOTOS.get(_norm_phone(identifier))
    if not raw:
        return None
    for magic in (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n"):
        idx = raw.find(magic, 0, 8)
        if idx != -1:
            return raw[idx:]
    return None


def name_map() -> dict:
    """Full {normalized_identifier: name} mapping."""
    _ensure_loaded()
    return dict(_NAMES)


if __name__ == "__main__":
    _ensure_loaded()
    print(f"Loaded {len(_NAMES)} named identifiers, {len(_PHOTOS)} with photos.")
    import sys
    for q in sys.argv[1:]:
        print(f"  {q:24} -> {resolve_name(q)}  (photo: {'yes' if get_photo(q) else 'no'})")
