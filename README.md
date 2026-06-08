# Message Analytics

**Explore your own iMessage/SMS history as a private, on-device relationship dashboard.**
Export conversations to Markdown and analyze emotional and behavioral signals over
time — sentiment, frequency, pursue–withdraw dynamics, Gottman's Four Horsemen,
emotional "who-leads" contagion, and trust-related language.

> 🔒 **100% local. Nothing ever leaves your machine.** No accounts, no cloud, no
> uploads. Your messages are read from the local macOS database in read-only mode.
> All personal data is git-ignored and never committed.

All charts below are generated from **synthetic sample data** (`generate_samples.py`) —
no real messages.

| Sentiment over time (per person) | Push–pull dynamics |
|---|---|
| ![sentiment](docs/sample_sentiment.png) | ![pushpull](docs/sample_pushpull.png) |
| **Conflict signals (Gottman)** | **Who leads the mood?** |
| ![gottman](docs/sample_gottman.png) | ![wholeads](docs/sample_wholeads.png) |

## What it measures

- **Per-person signals over time**: sentiment, % positive/negative, emoji rate,
  message length, reply latency, who-initiates, reciprocity, affection, questions,
  late-night share, media share, vulnerability.
- **Pursue–withdraw (demand–withdraw)**: a signed index showing who reaches out
  while the other pulls back (Christensen & Heavey, 1990).
- **Gottman metrics**: the Four Horsemen (criticism, contempt, defensiveness,
  stonewalling), repair attempts, harsh startup, and the 5:1 positivity ratio.
- **Who leads**: emotional contagion / lead–lag — whose mood the other mirrors.
- **Trust signals**: commitments, accountability, affirmation, distrust language.
- **All relationships** overview and an optional **family** view.

> ⚠️ **These are text proxies for reflection, not clinical diagnoses.** Sentiment
> uses VADER; behavioral metrics use transparent, documented heuristics. They show
> *patterns worth thinking about*, and cannot measure anyone's real trustworthiness
> or validate/invalidate anyone's feelings.

## Setup (macOS)

1. **Grant Full Disk Access** to your terminal / VS Code:
   System Settings → Privacy & Security → Full Disk Access → enable it →
   **fully quit (Cmd+Q) and reopen**. (Required to read `~/Library/Messages/chat.db`.)
2. Install deps: `pip install -r requirements.txt`
3. (Optional) Create `aliases.json` from the template to add names / exclusions:
   `cp aliases.example.json aliases.json` and edit.

## Usage

```bash
python3 export_chat.py --list                 # see who you talk to most
python3 export_chat.py --who alex@example.com --out alex.md
python3 extract.py                             # build the local cache + coverage report
streamlit run app.py                           # launch dashboard at localhost:8501
python3 generate_samples.py                    # render the synthetic sample charts
```

## Files

| File | Role |
|------|------|
| `messages_lib.py` | Read-only access to `chat.db` (snapshot, timestamps, `attributedBody` decode) |
| `contacts.py` | Resolve phone/email → names & photos from the macOS Contacts DB |
| `aliases.py` / `aliases.json` | Merge a person's identifiers, exclude noise, set names (gitignored) |
| `extract.py` | Load all messages into a tidy cached table |
| `signals.py` | Per-person signals over time |
| `dynamics.py` | Pursue–withdraw + who-leads contagion |
| `gottman.py` | Four Horsemen, repair, positivity ratio |
| `trust.py` | Trust-related language signals |
| `overview.py` | Cross-relationship comparison |
| `app.py` | Streamlit dashboard |

## Privacy & scope

The macOS `chat.db` reflects what's on *this* Mac — if Messages-in-iCloud history
or older devices aren't synced here, some history may be missing (`extract.py`
prints a coverage report). Nothing is uploaded; this is a personal, on-device tool.
