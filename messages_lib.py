"""
messages_lib.py — Core utilities for reading the macOS Messages database.

The Messages app stores everything in a local SQLite DB at
~/Library/Messages/chat.db. This module handles the gnarly parts:

  * Safely snapshotting the live DB (it has WAL/-shm sidecar files).
  * Apple's timestamp format (nanoseconds since 2001-01-01 UTC).
  * Decoding message text from the binary `attributedBody` blob, which is
    where modern macOS stores text when the plain `text` column is NULL.
  * Resolving handles (phone numbers / emails) and chat membership.

No third-party dependencies — standard library only.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# Apple epoch: 2001-01-01 00:00:00 UTC
APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)

DEFAULT_DB = os.path.expanduser("~/Library/Messages/chat.db")


def snapshot_db(src: str = DEFAULT_DB) -> str:
    """Copy the live chat.db (plus WAL/-shm sidecars) to a temp file.

    Messages keeps the DB open in WAL mode, so recent messages live in
    chat.db-wal until checkpointed. We copy all three so the snapshot is
    complete and we never touch the original.

    Returns the path to the temp copy. Caller is responsible for cleanup
    (or just let the OS clear /tmp).
    """
    tmpdir = tempfile.mkdtemp(prefix="msgexport_")
    dst = os.path.join(tmpdir, "chat.db")
    shutil.copy2(src, dst)
    for suffix in ("-wal", "-shm"):
        side = src + suffix
        if os.path.exists(side):
            shutil.copy2(side, dst + suffix)
    return dst


def connect(db_path: str) -> sqlite3.Connection:
    """Open a read-only connection to a chat.db snapshot."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def apple_time_to_dt(raw: int) -> datetime | None:
    """Convert a Messages `date` value to a timezone-aware datetime.

    Modern macOS stores nanoseconds since the Apple epoch; very old DBs
    stored seconds. Detect by magnitude.
    """
    if not raw:
        return None
    # Nanosecond values are ~10^18; second values are ~10^9.
    seconds = raw / 1e9 if raw > 1e11 else raw
    return APPLE_EPOCH + timedelta(seconds=seconds)


def decode_attributed_body(blob: bytes | None) -> str | None:
    """Extract plain text from a serialized NSAttributedString blob.

    This is the typedstream/NSArchiver format. We use the well-established
    heuristic: the message text is an inline byte string that follows the
    "NSString" class marker and a '+' (0x2b) byte, with a length prefix.
    Lengths >= 128 use a 0x81 + 2-byte-little-endian encoding.
    """
    if not blob:
        return None
    try:
        marker = blob.find(b"NSString")
        if marker == -1:
            return None
        plus = blob.find(b"\x2b", marker)  # '+' signals an inline string
        if plus == -1:
            return None
        i = plus + 1
        length = blob[i]
        if length == 0x81:  # extended length: next 2 bytes, little-endian
            length = int.from_bytes(blob[i + 1 : i + 3], "little")
            i += 3
        else:
            i += 1
        return blob[i : i + length].decode("utf-8", errors="replace")
    except Exception:
        return None


def message_text(row: sqlite3.Row) -> str | None:
    """Best-effort text for a message row: plain column first, then blob.

    Strips U+FFFC (object-replacement char) used as an attachment placeholder;
    returns None if nothing meaningful remains.
    """
    txt = row["text"] or decode_attributed_body(row["attributedBody"])
    if not txt:
        return None
    txt = txt.replace("￼", "").strip()
    return txt or None


@dataclass
class Chat:
    chat_id: int
    guid: str
    display_name: str
    identifiers: tuple[str, ...]  # phone numbers / emails in the chat
    message_count: int

    @property
    def label(self) -> str:
        if self.display_name:
            return self.display_name
        return ", ".join(self.identifiers) or self.guid


def list_chats(conn: sqlite3.Connection) -> list[Chat]:
    """Return all chats with their member identifiers and message counts."""
    rows = conn.execute(
        """
        SELECT c.ROWID AS chat_id,
               c.guid AS guid,
               COALESCE(c.display_name, '') AS display_name,
               GROUP_CONCAT(DISTINCT h.id) AS identifiers,
               COUNT(DISTINCT cmj.message_id) AS msg_count
        FROM chat c
        LEFT JOIN chat_handle_join chj ON chj.chat_id = c.ROWID
        LEFT JOIN handle h ON h.ROWID = chj.handle_id
        LEFT JOIN chat_message_join cmj ON cmj.chat_id = c.ROWID
        GROUP BY c.ROWID
        ORDER BY msg_count DESC
        """
    ).fetchall()
    chats = []
    for r in rows:
        ids = tuple((r["identifiers"] or "").split(",")) if r["identifiers"] else ()
        chats.append(
            Chat(
                chat_id=r["chat_id"],
                guid=r["guid"],
                display_name=r["display_name"],
                identifiers=ids,
                message_count=r["msg_count"],
            )
        )
    return chats


def normalize_phone(s: str) -> str:
    """Reduce a phone/email identifier to digits (or lowercased email) for matching."""
    s = s.strip().lower()
    if "@" in s:
        return s
    return "".join(ch for ch in s if ch.isdigit())


def find_chats_for_identifier(conn: sqlite3.Connection, identifier: str) -> list[Chat]:
    """Find chats whose members match a phone number or email.

    Matches loosely: phone numbers compared by trailing digits, emails by
    exact (case-insensitive) match.
    """
    target = normalize_phone(identifier)
    matches = []
    for chat in list_chats(conn):
        for member in chat.identifiers:
            m = normalize_phone(member)
            if "@" in target:
                if m == target:
                    matches.append(chat)
                    break
            else:
                # match if either is a suffix of the other (handles +1 prefixes)
                if m and (m.endswith(target) or target.endswith(m)):
                    matches.append(chat)
                    break
    return matches


def iter_messages(conn: sqlite3.Connection, chat_id: int):
    """Yield messages for a chat in chronological order.

    Each yielded dict has: dt (datetime), is_from_me (bool), sender (str),
    text (str), service (iMessage/SMS).
    """
    rows = conn.execute(
        """
        SELECT m.ROWID, m.text, m.attributedBody, m.date,
               m.is_from_me, m.service,
               COALESCE(h.id, '') AS sender_id
        FROM message m
        JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
        LEFT JOIN handle h ON h.ROWID = m.handle_id
        WHERE cmj.chat_id = ?
        ORDER BY m.date ASC
        """,
        (chat_id,),
    )
    for r in rows:
        yield {
            "dt": apple_time_to_dt(r["date"]),
            "is_from_me": bool(r["is_from_me"]),
            "sender": r["sender_id"],
            "text": message_text(r),
            "service": r["service"],
        }
