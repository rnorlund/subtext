# Message Analytics

Export and analyze your macOS Messages — fully local, nothing leaves your machine.

## Setup (one-time)

1. **Grant Full Disk Access** to whatever app runs these scripts
   (VS Code or your terminal): System Settings → Privacy & Security →
   Full Disk Access → add & enable it → **fully quit (Cmd+Q) and reopen**.
2. Dependencies (already installed here): `pandas`, `plotly`, `streamlit`,
   `vaderSentiment`.

## Usage

```bash
# 1. See who you talk to most
python3 export_chat.py --list

# 2. Export one conversation to Markdown
python3 export_chat.py --who serieliz@gmail.com --out sarah.md --me "Me" --them "Sarah"

# 3. Build the all-contacts cache + print a coverage summary
python3 extract.py

# 4. Launch the dashboard (opens http://localhost:8501)
streamlit run app.py
```

## Files

| File | Role |
|------|------|
| `messages_lib.py` | Read-only access to `chat.db`: snapshot, timestamps, `attributedBody` decoding, handle resolution |
| `export_chat.py`  | Export a single conversation to Markdown |
| `extract.py`      | Load **all** messages into a tidy cached table (`messages.parquet`) |
| `signals.py`      | Compute the 13 relationship signals as time series |
| `app.py`          | Streamlit web dashboard |

## The 13 signals (all plotted over time)

volume · net sentiment · % positive · % negative · emoji rate · avg words ·
reciprocity · initiation share · reply latency · question rate · affection ·
late-night share · media share

Sentiment uses **VADER** (MIT license) — safe to ship in a paid product.

## Note on completeness

The Mac `chat.db` may not equal your iPhone. SMS (green-bubble) texts and
history from before the Mac was set up can be iPhone/iCloud-only. `extract.py`
prints a coverage summary so you can judge. For a guaranteed-complete source,
an unencrypted local iPhone backup contains its own `sms.db` we can parse.
