#!/usr/bin/env python3
"""
export_chat.py — Export a Messages conversation to Markdown.

Usage:
    # See who you talk to (sorted by volume) so you can pick a target:
    python3 export_chat.py --list

    # Export by phone/email (auto-finds the chat):
    python3 export_chat.py --who alex@example.com --out alex.md
    python3 export_chat.py --who 555-123-4567 --out alex.md

Reads a read-only snapshot of ~/Library/Messages/chat.db. Never modifies
the original database.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import messages_lib as ml

_MIN_DT = datetime.min.replace(tzinfo=timezone.utc)


def cmd_list(conn) -> None:
    chats = ml.list_chats(conn)
    print(f"{'msgs':>8}  {'name / members'}")
    print("-" * 60)
    for c in chats[:80]:
        print(f"{c.message_count:>8}  {c.label}")
    print(f"\n{len(chats)} chats total. Use --who <phone-or-email> to export one.")


def export_markdown(conn, chats: list[ml.Chat], out_path: str, me_name: str, them_name: str) -> None:
    """Export one or more 1:1 threads, merged chronologically into one file."""
    # Gather messages from every supplied thread, then sort by time.
    merged = []
    for chat in chats:
        for msg in ml.iter_messages(conn, chat.chat_id):
            merged.append(msg)
    merged.sort(key=lambda m: (m["dt"] or _MIN_DT, m["text"] or ""))
    # Drop exact duplicates (same time + text + direction) in case threads overlap.
    seen = set()
    deduped = []
    for m in merged:
        key = (m["dt"], m["is_from_me"], m["text"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(m)

    lines: list[str] = []
    lines.append(f"# Messages with {them_name}")
    lines.append("")
    threads = "; ".join(c.label for c in chats)
    lines.append(f"- Threads merged: {threads}")
    lines.append(f"- Total messages: {len(deduped)}")
    lines.append("")

    current_day = None
    count = 0
    empty = 0
    for msg in deduped:
        dt = msg["dt"]
        text = msg["text"]
        if not text:
            empty += 1
            continue
        if dt and dt.date() != current_day:
            current_day = dt.date()
            lines.append("")
            lines.append(f"## {current_day:%A, %B %d, %Y}")
            lines.append("")
        who = me_name if msg["is_from_me"] else them_name
        stamp = f"{dt:%I:%M %p}".lstrip("0") if dt else "??:??"
        # Indent multi-line messages cleanly as a blockquote.
        body = text.replace("\n", "\n> ")
        lines.append(f"**{who}** · {stamp}")
        lines.append(f"> {body}")
        lines.append("")
        count += 1

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Wrote {count} messages to {out_path}")
    if empty:
        print(f"   ({empty} messages had no decodable text — usually attachments/reactions.)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true", help="list all chats by volume")
    p.add_argument("--who", help="phone number or email to export")
    p.add_argument("--out", default="conversation.md", help="output .md path")
    p.add_argument("--me", default="Me", help="your name in the transcript")
    p.add_argument("--them", help="their name in the transcript (default: the identifier)")
    p.add_argument("--db", default=ml.DEFAULT_DB, help="path to chat.db")
    args = p.parse_args()

    try:
        snap = ml.snapshot_db(args.db)
    except PermissionError:
        print(
            "❌ Operation not permitted reading chat.db.\n"
            "   Grant Full Disk Access to your terminal/VS Code in\n"
            "   System Settings → Privacy & Security → Full Disk Access,\n"
            "   then fully quit (Cmd+Q) and reopen it.",
            file=sys.stderr,
        )
        return 1

    conn = ml.connect(snap)

    if args.list:
        cmd_list(conn)
        return 0

    if not args.who:
        print("Specify --list or --who <phone-or-email>.", file=sys.stderr)
        return 2

    # Allow several identifiers (comma-separated) so split email/phone threads merge.
    idents = [s.strip() for s in args.who.split(",") if s.strip()]
    matches = []
    for ident in idents:
        matches.extend(ml.find_chats_for_identifier(conn, ident))
    # Keep only 1:1 threads (exclude group chats), de-duplicated by chat_id.
    one_to_one = {c.chat_id: c for c in matches if len(c.identifiers) <= 1}
    if not one_to_one:
        print(f"No 1:1 chat found matching '{args.who}'. Try --list.", file=sys.stderr)
        return 1
    chats = sorted(one_to_one.values(), key=lambda c: c.message_count, reverse=True)
    them = args.them or chats[0].label
    if len(chats) > 1:
        labels = ", ".join(f"{c.label} ({c.message_count})" for c in chats)
        print(f"Merging {len(chats)} 1:1 threads: {labels}")
    export_markdown(conn, chats, args.out, args.me, them)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
