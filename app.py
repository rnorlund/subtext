#!/usr/bin/env python3
"""
app.py — Local web dashboard for your message analytics.

Run it with:
    streamlit run app.py

Opens at http://localhost:8501. Everything runs on your machine; no data
leaves the computer.

Pick a contact, choose a time granularity, and explore all 13 relationship
signals plotted over time. Built on the cached table from extract.py.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DEMO = os.environ.get("MSGANALYTICS_DEMO") == "1"

import extract
import signals as sig
import dynamics as dyn
import overview
import contacts
import gottman
import aliases
import trust
import personality

st.set_page_config(page_title="Message Analytics", layout="wide")


@st.cache_data(show_spinner="Loading messages…")
def get_data(rebuild: bool) -> pd.DataFrame:
    if DEMO:                                  # public preview on synthetic data
        import demo_data
        return demo_data.demo_dataframe()
    # Merge aliased identifiers (a person's email + phone) into one person.
    return aliases.apply_aliases(extract.load_cached(rebuild=rebuild))


def _children_list():
    if DEMO:
        import demo_data
        return demo_data.CHILDREN
    return aliases.children()


def _main_list():
    if DEMO:
        import demo_data
        return ["Alex (partner)", "Mom", "Kid A"]
    return aliases.main_people()


def _my_name():
    if DEMO:
        import demo_data
        return demo_data.ME_NAME
    return aliases.me_name()


def _dense_start(sub: pd.DataFrame):
    """First date where the conversation becomes sustained/dense.

    Trims leading sparse-then-gappy history (e.g. a handful of 2024 messages
    before the thread really gets going) so every chart starts where the data
    is real, not on isolated early points.
    """
    if sub.empty:
        return None
    vol = sub.set_index("dt").resample("W").size()
    nz = vol[vol > 0]
    if len(nz) < 6:
        return None
    floor = max(3, 0.25 * nz.median())
    ok = (vol >= floor).values
    for i in range(len(ok)):
        # first week that's dense AND part of a sustained dense run (3 of next 4).
        if ok[i] and ok[i:i + 4].sum() >= 3:
            return vol.index[i]
    return None


def _subset(contact: str, months: int, rebuild: bool) -> pd.DataFrame:
    """1:1 messages for a contact: trimmed to the dense period, then last N months."""
    df = get_data(rebuild)
    sub = df[(df["contact"] == contact) & (~df["is_group"])]
    ds = _dense_start(sub)              # drop sparse leading history
    if ds is not None:
        sub = sub[sub["dt"] >= ds]
    if months and not sub.empty:
        cutoff = sub["dt"].max() - pd.DateOffset(months=months)
        sub = sub[sub["dt"] >= cutoff]
    return sub


@st.cache_data(show_spinner="Computing signals…")
def get_signals(contact: str, freq: str, months: int, rebuild: bool) -> pd.DataFrame:
    return sig.compute_signals(_subset(contact, months, rebuild), freq=freq)


@st.cache_data(show_spinner="Splitting signals by person…")
def get_signals_split(contact: str, freq: str, months: int, rebuild: bool) -> dict:
    return sig.compute_signals_split(_subset(contact, months, rebuild), freq=freq)


@st.cache_data(show_spinner="Computing pursue–withdraw dynamics…")
def get_pursuit(contact: str, freq: str, months: int, rebuild: bool) -> pd.DataFrame:
    return dyn.compute_pursuit(_subset(contact, months, rebuild), freq=freq)


@st.cache_data(show_spinner="Computing push–pull…")
def get_push_pull(contact: str, freq: str, months: int, rebuild: bool) -> pd.DataFrame:
    return dyn.compute_push_pull(_subset(contact, months, rebuild), freq=freq)


@st.cache_data(show_spinner="Analyzing who leads…")
def get_who_leads(contact: str, freq: str, months: int, rebuild: bool) -> dict:
    return dyn.who_leads(_subset(contact, months, rebuild), freq=freq)


@st.cache_data(show_spinner="Computing Gottman metrics…")
def get_gottman(contact: str, freq: str, months: int, rebuild: bool) -> dict:
    return gottman.compute_gottman(_subset(contact, months, rebuild), freq=freq)


@st.cache_data(show_spinner="Gottman summary…")
def get_gottman_summary(contact: str, months: int, rebuild: bool) -> dict:
    return gottman.summary(_subset(contact, months, rebuild))


@st.cache_data(show_spinner="Computing trust signals…")
def get_trust(contact: str, freq: str, months: int, rebuild: bool) -> dict:
    return trust.compute_trust(_subset(contact, months, rebuild), freq=freq)


@st.cache_data(show_spinner="Calibrating personality baseline…")
def get_personality_baseline(rebuild: bool) -> dict:
    return personality.population(get_data(rebuild))


@st.cache_data(show_spinner="Estimating personality…")
def get_personality(contact: str, months: int, rebuild: bool) -> dict:
    return personality.analyze(_subset(contact, months, rebuild),
                               baseline=get_personality_baseline(rebuild))


@st.cache_data
def load_icons() -> dict:
    """Slice icons.png (4×2 grid) into per-section square PNGs, trimmed & centered."""
    import os
    from io import BytesIO
    from PIL import Image
    if not os.path.exists("icons.png"):
        return {}
    im = Image.open("icons.png").convert("RGBA")
    W, H = im.size
    cols, rows = 4, 2
    cw, ch = W // cols, H // rows
    order = ["emotional", "activity", "connection", "pushpull",
             "gottman", "wholeads", "trust", "personality"]
    out = {}
    for idx, key in enumerate(order):
        r, c = divmod(idx, cols)
        cell = im.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch))
        bbox = cell.getbbox()
        if not bbox:          # empty cell (e.g. no personality icon yet) — skip
            continue
        cell = cell.crop(bbox)
        s = max(cell.size) + 16
        canvas = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        canvas.paste(cell, ((s - cell.size[0]) // 2, (s - cell.size[1]) // 2), cell)
        buf = BytesIO()
        canvas.save(buf, "PNG")
        out[key] = buf.getvalue()
    return out


@st.cache_data(show_spinner="Summarizing all relationships…")
def get_overview(rebuild: bool) -> pd.DataFrame:
    return overview.summarize(get_data(rebuild))


# Two-color scheme for per-person splits — bright, to pop on a dark background.
COLOR_ME = "#3aa0ff"      # you — vivid sky blue
COLOR_THEM = "#ff5fa2"    # them — vivid pink


st.title("💬 Message Analytics")
if DEMO:
    st.success("🧪 **DEMO MODE** — showing synthetic, deidentified sample data "
               "(no real messages). This is the public preview.")
else:
    st.caption("100% local. Nothing leaves your machine.")

with st.sidebar:
    st.header("Controls")
    rebuild = st.button("🔄 Rebuild from Messages DB")
    try:
        df = get_data(rebuild)
    except PermissionError:
        st.error(
            "Can't read chat.db — grant **Full Disk Access** to VS Code / your "
            "terminal in System Settings → Privacy & Security, then fully quit "
            "(Cmd+Q) and reopen."
        )
        st.stop()

    if df.empty:
        st.warning("No messages found.")
        st.stop()

    # Rank 1:1 contacts by volume.
    counts = (
        df[~df["is_group"]]
        .groupby("contact")
        .size()
        .sort_values(ascending=False)
    )

    # Resolve each identifier to a real contact name once.
    def disp_name(c: str) -> str:
        if aliases.is_alias(c):       # already a canonical person name
            return c
        return contacts.resolve_name(c) or str(c)

    view_opts = ["Single relationship", "All relationships"]
    if _children_list():
        view_opts.append("👨‍👩‍👧 Family (kids)")
    mode = st.radio("View", view_opts, index=0)
    contact = st.selectbox(
        "Contact",
        counts.index.tolist(),
        format_func=lambda c: f"{disp_name(c)}  ({counts[c]:,} msgs)",
        disabled=(mode != "Single relationship"),
    )
    freq_label = st.radio("Granularity", ["Daily", "Weekly", "Monthly"], index=1)
    freq = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}[freq_label]
    st.divider()
    me_name = st.text_input("Your name", value=_my_name())
    them_name = st.text_input("Their name", value=disp_name(contact))

# Coverage summary up top.
with st.expander("📊 Data coverage (read me — checks for iPhone-only gaps)"):
    st.code(extract.coverage_summary(df))

# ── All-relationships overview ────────────────────────────────────────────
if mode == "All relationships":
    st.subheader("🏆 All relationships — where you're thriving & where to invest")
    ov = get_overview(rebuild)
    if ov.empty:
        st.warning("Not enough data.")
        st.stop()

    ov = ov.copy()
    ov.index = [disp_name(c) for c in ov.index]

    # Filters so acquaintances don't crowd the charts: a core/family list and/or
    # an adjustable minimum-message threshold.
    _main = _main_list()
    fcol = st.columns([2, 3])
    with fcol[0]:
        only_main = st.checkbox("⭐ Main people only (family/core)", value=bool(_main)) if _main else False
    with fcol[1]:
        max_m = int(ov["messages"].max())
        min_msgs = st.slider("Minimum messages", 20, min(2000, max_m), min(100, max_m),
                             step=10, disabled=only_main)
    if only_main:
        present = [m for m in _main if m in ov.index]
        if present:
            ov = ov.loc[present]
    else:
        ov = ov[ov["messages"] >= min_msgs]
    if ov.empty:
        st.warning("No contacts match the current filter.")
        st.stop()
    st.caption(f"Showing **{len(ov)}** "
               + ("core people." if only_main else f"contacts with ≥ {min_msgs} messages."))
    disp = ov[list(overview.DISPLAY_COLS.keys())].rename(columns=overview.DISPLAY_COLS)
    st.dataframe(
        disp.style.format({
            "Your share": "{:.0%}", "Net sentiment": "{:+.2f}",
            "% positive": "{:.0%}", "% negative": "{:.0%}",
            "Pos:Neg ratio": "{:.1f}", "Affection": "{:.2f}",
            "Emoji/msg": "{:.2f}", "Vulnerability": "{:.3f}",
        }).background_gradient(subset=["Net sentiment", "Pos:Neg ratio"], cmap="RdYlGn"),
        use_container_width=True, height=520,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Warmth vs. activity** — bubble size = messages")
        # Only label the top dozen by volume so text doesn't overlap; rest = hover.
        topn = set(ov.sort_values("messages", ascending=False).head(12).index)
        labels = [str(c) if c in topn else "" for c in ov.index]
        fig = go.Figure(go.Scatter(
            x=ov["messages"], y=ov["net_sentiment"], mode="markers+text",
            text=labels, textposition="top center", textfont=dict(size=10),
            hovertext=[str(c) for c in ov.index], hoverinfo="text+x+y",
            marker=dict(size=(ov["messages"] ** 0.5) / 2 + 4, color=ov["net_sentiment"],
                        colorscale="RdYlGn", showscale=True, cmin=-0.3, cmax=0.6,
                        line=dict(width=0.5, color="rgba(255,255,255,0.3)")),
        ))
        fig.update_layout(height=460, xaxis_title="messages (log)", xaxis_type="log",
                          yaxis_title="net sentiment", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Only the 12 highest-volume contacts are labeled; hover for the rest.")
    with c2:
        st.markdown("**Most & least positive**")
        ranked = ov.sort_values("net_sentiment")
        tail = ranked if len(ranked) <= 16 else pd.concat([ranked.head(8), ranked.tail(8)])
        # Force categorical y — otherwise phone-number-like labels become a numeric axis.
        ylabels = [str(c) for c in tail.index]
        bar = go.Figure(go.Bar(
            x=tail["net_sentiment"].values, y=ylabels, orientation="h",
            marker=dict(color=tail["net_sentiment"].values, colorscale="RdYlGn", cmin=-0.3, cmax=0.6),
        ))
        bar.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="net sentiment",
                          yaxis=dict(type="category", categoryorder="array",
                                     categoryarray=ylabels, autorange="reversed"))
        st.plotly_chart(bar, use_container_width=True)

    st.caption(
        "Sentiment via VADER. 'Days since last' flags relationships going quiet. "
        "Switch to **Single relationship** in the sidebar for deep-dive charts."
    )
    st.stop()

# ── Family: do the kids mirror your mood? ─────────────────────────────────
if mode == "👨‍👩‍👧 Family (kids)":
    me = aliases.me_name()
    st.subheader(f"👨‍👩‍👧 Do the kids mirror {me}'s mood?")
    st.info(
        "For each child's 1:1 thread, this measures **emotional contagion**: when "
        f"{me} sends an emotional message, how often the child's next reply carries the "
        "same emotion — plus a lead–lag correlation of sentiment over time. "
        "**Correlation, not proof of causation** — kids' moods have countless causes "
        "beyond texts. A reflection tool, not a verdict."
    )
    kids = [k for k in _children_list() if k in counts.index]
    if not kids:
        st.warning("No configured children found in the data.")
        st.stop()

    rows = []
    for kid in kids:
        ksub = df[(df["contact"] == kid) & (~df["is_group"])]
        wl = dyn.who_leads(ksub, freq="ME")
        cc = wl.get("crosscorr", {}) if wl else {}
        peak = max(cc, key=lambda k: cc[k]) if cc else 0
        rows.append({
            "kid": kid, "msgs": len(ksub),
            "neg_mirror": wl.get("neg_me_to_them", 0) if wl else 0,
            "pos_mirror": wl.get("pos_me_to_them", 0) if wl else 0,
            "lag": peak, "corr": cc.get(peak, 0) if cc else 0,
        })

    # Comparison bars: how much each kid mirrors your negativity / positivity.
    fig = go.Figure()
    fig.add_trace(go.Bar(name=f"mirrors {me}'s negativity",
                         x=[r["kid"] for r in rows], y=[100 * r["neg_mirror"] for r in rows],
                         marker_color="#E74C3C",
                         text=[f"{100*r['neg_mirror']:.0f}%" for r in rows], textposition="outside"))
    fig.add_trace(go.Bar(name=f"mirrors {me}'s positivity",
                         x=[r["kid"] for r in rows], y=[100 * r["pos_mirror"] for r in rows],
                         marker_color="#2ca02c",
                         text=[f"{100*r['pos_mirror']:.0f}%" for r in rows], textposition="outside"))
    fig.update_layout(barmode="group", height=400, yaxis_title="% of replies that mirror your mood",
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.08, x=0),
                      title=dict(text="When you send an emotional message, how often each kid echoes it",
                                 font=dict(size=14)))
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(len(rows))
    for col, r in zip(cols, rows):
        most = "negativity" if r["neg_mirror"] >= r["pos_mirror"] else "positivity"
        lead = (f"your mood tends to **precede** {r['kid']}'s by {abs(r['lag'])} period(s)"
                if r["lag"] > 0 else
                f"{r['kid']}'s mood tends to **precede** yours" if r["lag"] < 0 else
                "moods move **in sync**")
        col.metric(r["kid"], f"{r['msgs']:,} msgs",
                   f"echoes your {most} most", delta_color="off")
        col.caption(f"Lead–lag: {lead} (r={r['corr']:+.2f}).")

    st.caption(
        "Higher bars = that child's replies more often match the emotion you just sent. "
        "Use it to notice patterns worth a gentle conversation — not to assign blame."
    )
    st.stop()

# Header: contact photo + name.
def _decode_photo(raw: bytes):
    """Return PNG bytes PIL/Streamlit can render, or None if undecodable."""
    if not raw:
        return None
    try:
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(raw))
        img.load()
        buf = BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


photo = _decode_photo(contacts.get_photo(aliases.primary_identifier(contact)))
hcol = st.columns([1, 9])
with hcol[0]:
    if photo:
        st.image(photo, width=72)
    else:
        st.markdown(
            f"<div style='width:72px;height:72px;border-radius:50%;background:#1f77b4;"
            f"display:flex;align-items:center;justify-content:center;font-size:30px;"
            f"color:white;'>{(them_name or '?')[0].upper()}</div>",
            unsafe_allow_html=True,
        )
with hcol[1]:
    st.markdown(f"### {them_name}")
    st.caption(f"{contact}")

# ── Controls: time range + outlier toggle ─────────────────────────────────
tc = st.columns([5, 2])
with tc[0]:
    range_label = st.radio(
        "Time range", ["Last 6 months", "Last 12 months", "All time"],
        index=2, horizontal=True)
    months = {"Last 6 months": 6, "Last 12 months": 12, "All time": 0}[range_label]
with tc[1]:
    REMOVE_OUTLIERS = st.checkbox("Remove outliers", value=True)

sub = _subset(contact, months, rebuild)
if sub.empty or len(sub) < 5:
    st.warning("Not enough messages in this time range to chart.")
    st.stop()

sigs = get_signals(contact, freq, months, rebuild)
split = get_signals_split(contact, freq, months, rebuild)

# Headline metrics.
c1, c2, c3, c4 = st.columns(4)
c1.metric("Messages", f"{len(sub):,}")
c2.metric(f"{me_name} sent", f"{100*sub['is_from_me'].mean():.0f}%")
c3.metric("Span", f"{(sub['dt'].max() - sub['dt'].min()).days:,} days")
c4.metric("Avg net sentiment", f"{sig.enrich(sub)['compound'].mean():+.2f}")


def _drop_outliers(x_dt, y):
    """IQR outlier removal (1.5×), preserving NaNs, when the toggle is on."""
    ys = pd.Series(list(y))
    if not REMOVE_OUTLIERS or ys.notna().sum() < 4:
        return list(x_dt), list(ys)
    q1, q3 = ys.quantile(0.25), ys.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    keep = ys.between(lo, hi) | ys.isna()
    xk = [x for x, k in zip(list(x_dt), keep) if k]
    yk = [v for v, k in zip(list(ys), keep) if k]
    return xk, yk


def _add_scatter_with_fit(fig, x_dt, y, name, color, drop=True):
    """Add a set of dots plus its linear best-fit line.

    drop=False skips outlier removal — important for rare-event/count series
    (e.g. Four Horsemen) where the rare spikes ARE the signal.
    """
    if drop:
        x_dt, ylist = _drop_outliers(x_dt, y)
    else:
        ylist = list(y)
    y = pd.Series(ylist, index=range(len(ylist)))
    mask = y.notna().values
    if mask.sum() == 0:
        return
    fig.add_trace(go.Scatter(
        x=x_dt, y=list(y), mode="markers", name=name,
        marker=dict(size=8, color=color, opacity=0.95,
                    line=dict(width=1, color="rgba(255,255,255,0.85)"))))
    if mask.sum() >= 2:
        xnum = np.array([pd.Timestamp(t).toordinal() for t in x_dt])[mask]
        yv = y.values[mask].astype(float)
        slope, intercept = np.polyfit(xnum, yv, 1)
        xline = np.array([xnum.min(), xnum.max()])
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp.fromordinal(int(v)) for v in xline],
            y=slope * xline + intercept, mode="lines",
            line=dict(width=2.5, color=color), showlegend=False, hoverinfo="skip"))


def _safe_rate(count: pd.Series, vol: pd.Series) -> pd.Series:
    """Windowed per-100-msg rate that ignores sparse bins and partial windows.

    Drops bins with too few messages (one-message spikes) and requires a FULL
    rolling window, so the line only appears where there's enough real data —
    no misleading straight lines across near-empty early/late periods.
    """
    if count.empty:
        return count
    med = vol[vol > 0].median() if (vol > 0).any() else 0
    floor = max(3, 0.25 * (med or 0))
    keep = vol >= floor
    count, vol = count[keep], vol[keep]
    if len(count) < 3:
        return pd.Series(dtype=float)
    w = max(3, len(count) // 6)
    rate = 100 * count.rolling(w, min_periods=w).sum() / vol.rolling(w, min_periods=w).sum()
    return rate.dropna()


def _chart(fig, label, height=260, legend=False):
    fig.update_layout(
        title=dict(text=label, font=dict(size=15)),
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        showlegend=legend, xaxis_title=None, yaxis_title=None,
        legend=dict(orientation="h", y=1.0, x=0))
    return fig


def _selected_x(event):
    """Extract clicked x-values from a Streamlit plotly_chart selection event."""
    try:
        pts = event["selection"]["points"]
    except Exception:
        pts = getattr(getattr(event, "selection", None), "points", None) or []
    return [p.get("x") for p in pts if p.get("x") is not None]


def _messages_in_period(x, freq):
    """Return the messages whose period-bin matches the clicked x value."""
    key = pd.Timestamp(x)
    s = sub.set_index("dt")
    groups = list(s.groupby(pd.Grouper(freq=freq)))
    best, best_gap = None, None
    for label, gdf in groups:
        gap = abs((pd.Timestamp(label) - key).total_seconds())
        if best_gap is None or gap < best_gap:
            best, best_gap = gdf, gap
    return best.reset_index() if best is not None else pd.DataFrame()


def _show_inspector(event, freq, key=None, flag=None):
    """If a point was clicked, show the messages in that exact bin — sorted so the
    ones that DROVE the clicked metric are at the top."""
    xs = _selected_x(event)
    if not xs:
        return
    gdf = _messages_in_period(xs[0], freq)
    if gdf.empty:
        return
    e = sig.enrich(gdf)
    if flag is not None:                       # only the flagged conflict messages
        import gottman as _gm
        e = _gm.enrich_gottman(gdf)
        e = e[e[flag] > 0]
    e = e.assign(who=e["is_from_me"].map({True: me_name, False: them_name}))
    e["sentiment"] = e["compound"].round(2)
    # Sort by what drove the clicked signal.
    if key == "pct_negative":
        e = e.sort_values("compound")                     # most negative first
        drv = "most-negative messages first"
    elif key == "pct_positive":
        e = e.sort_values("compound", ascending=False)    # most positive first
        drv = "most-positive messages first"
    elif key in ("emoji_rate",):
        e = e.sort_values("n_emoji", ascending=False)
        drv = "most emoji first"
    elif key in ("avg_words", "volume"):
        e = e.sort_values("n_words", ascending=False)
        drv = "longest messages first"
    else:
        e = e.reindex(e["compound"].abs().sort_values(ascending=False).index)
        drv = "biggest sentiment drivers first"
    view = e[["dt", "who", "text", "sentiment"]].rename(columns={"dt": "when"})
    span_a, span_b = gdf["dt"].min(), gdf["dt"].max()
    label = (f"{span_a:%b %d}–{span_b:%b %d, %Y}" if span_a.date() != span_b.date()
             else f"{span_a:%b %d, %Y}")
    with st.expander(
        f"🔍 {len(view)} messages in this point ({label}) — {drv}",
        expanded=True,
    ):
        st.caption("This dot is the **average** over these messages; the top rows moved it most. "
                   "Switch granularity to **Daily** in the sidebar for tighter, near-single-message points.")
        st.dataframe(view, use_container_width=True, height=340, hide_index=True)


def _person_series(fig, me_s, them_s, combined_s=None, scatter=True):
    """Add traces for me / them / both / combined per the global VIEW_MODE."""
    def emit(series, name, color):
        if series is None or len(pd.Series(series).dropna()) == 0:
            return
        if scatter:
            _add_scatter_with_fit(fig, series.index, series.values, name, color)
        else:
            fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines",
                                     line=dict(width=2.6, color=color), name=name))
    if VIEW_MODE == "me":
        emit(me_s, me_name, COLOR_ME)
    elif VIEW_MODE == "them":
        emit(them_s, them_name, COLOR_THEM)
    elif VIEW_MODE == "combined":
        if combined_s is None and me_s is not None and them_s is not None:
            combined_s = me_s.add(them_s, fill_value=0)
        emit(combined_s if combined_s is not None else me_s, "Combined", "#cfcfe0")
    else:  # both
        emit(me_s, me_name, COLOR_ME)
        emit(them_s, them_name, COLOR_THEM)


def render_grid(keys: list[str]) -> None:
    cols = st.columns(2)
    drawn = 0
    for key in keys:
        if key not in sigs:
            continue
        label, desc = sig.SIGNAL_META[key]
        fig = go.Figure()
        if key in sig.PER_PERSON_SIGNALS and not split["me"].empty:
            sm = split["me"][key] if key in split["me"] else None
            sthem = split["them"][key] if key in split["them"] else None
            _person_series(fig, sm, sthem, combined_s=sigs[key], scatter=True)
            legend = True
        else:
            series = sigs[key].dropna()
            _add_scatter_with_fit(fig, series.index, series.values, label, "#3aa0ff")
            legend = False
        _chart(fig, label, legend=legend)
        with cols[drawn % 2]:
            ev = st.plotly_chart(fig, use_container_width=True,
                                 on_select="rerun", key=f"grid_{key}")
            st.caption(desc + "  ·  *click a dot to read those messages*")
            _show_inspector(ev, freq, key=key)
        drawn += 1


# ── Section selector: one group at a time ─────────────────────────────────
GROUPS = {
    "💚 Emotional tone": ["net_sentiment", "pct_positive", "pct_negative",
                          "positivity_ratio", "emotional_balance"],
    "📊 Activity & cadence": ["volume", "reciprocity", "initiation_share",
                             "reply_latency_min", "late_night_share", "media_share"],
    "🤝 Connection & affection": ["affection", "encouragement", "emoji_rate",
                                 "question_rate", "avg_words", "vulnerability"],
}
SPECIAL = ["🧲 Pursue–withdraw (push–pull)", "🔬 Gottman (Four Horsemen)",
           "🧭 Who leads (causality)", "🤝 Trust signals", "🧠 Personality"]

st.divider()
st.markdown("**Section**")
_sections = list(GROUPS.keys()) + SPECIAL
# Unified emoji-in-button labels (consistent, icon inside the button).
_labels = ["💚 Emotional", "📊 Activity", "🤝 Connection", "🧲 Pursue–withdraw",
           "🔬 Gottman", "🧭 Who leads", "🛡️ Trust", "🧠 Personality"]
if st.session_state.get("section") not in _sections:
    st.session_state["section"] = _sections[0]
_ncol = 4
for _row in range(0, len(_sections), _ncol):
    _cols = st.columns(_ncol, gap="small")
    for _j in range(_ncol):
        _i = _row + _j
        if _i >= len(_sections):
            break
        with _cols[_j]:
            _active = st.session_state["section"] == _sections[_i]
            if st.button(_labels[_i], key=f"secbtn_{_i}", use_container_width=True,
                         type="primary" if _active else "secondary"):
                st.session_state["section"] = _sections[_i]
section = st.session_state["section"]

# Per-person view mode (applies to all per-person charts).
_view_opts = {
    f"Both — {me_name} & {them_name}": "both",
    f"{me_name} only": "me",
    f"{them_name} only": "them",
    "Combined (one line)": "combined",
}
_view_label = st.radio("Show", list(_view_opts.keys()), horizontal=True, index=0)
VIEW_MODE = _view_opts[_view_label]

st.caption(f"🔵 {me_name}   🔴 {them_name}   ·   dots = each period, line = best-fit trend"
           + ("   ·   outliers removed" if REMOVE_OUTLIERS else ""))

# ---- Signal groups ----
if section in GROUPS:
    render_grid(GROUPS[section])
    # Explicit emotional-vs-practical two-line chart in the Emotional tone group.
    if section.endswith("Emotional tone") and {"emotional_share", "practical_share"} <= set(sigs.columns):
        emo, prac = sigs["emotional_share"].dropna(), sigs["practical_share"].dropna()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=emo.index, y=100 * emo.values, mode="lines+markers",
                                 line=dict(width=2.6, color="#ff6f91"), marker=dict(size=5),
                                 name="Emotional"))
        fig.add_trace(go.Scatter(x=prac.index, y=100 * prac.values, mode="lines+markers",
                                 line=dict(width=2.6, color="#4cc9f0"), marker=dict(size=5),
                                 name="Practical"))
        _chart(fig, "Emotional vs. practical messages (% of messages)", height=320, legend=True)
        fig.update_layout(yaxis_title="% of messages")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Emotional** = strong sentiment or affection; **Practical** = logistics/"
                   "coordination (plans, errands, money, times). When the pink line sits above "
                   "the blue, the conversation is more feelings-driven than task-driven.")

        # Top 10 most positive / most negative individual messages.
        st.markdown("#### 💬 Strongest messages (by sentiment)")
        e = sig.enrich(sub)
        e = e[e["text"].str.len() > 3].copy()
        e["who"] = e["is_from_me"].map({True: me_name, False: them_name})
        view = e[["dt", "who", "text", "compound"]].rename(columns={"dt": "when", "compound": "score"})
        pc, nc = st.columns(2)
        with pc:
            st.markdown("**🟢 Top 10 most positive**")
            st.dataframe(view.nlargest(10, "score").round({"score": 2}),
                         hide_index=True, use_container_width=True, height=380)
        with nc:
            st.markdown("**🔴 Top 10 most negative**")
            st.dataframe(view.nsmallest(10, "score").round({"score": 2}),
                         hide_index=True, use_container_width=True, height=380)
        st.caption("Ranked by per-message VADER score (−1 to +1). Great for seeing the actual "
                   "high and low moments behind the trends.")

# ---- Pursue–withdraw (push–pull) ----
elif section == SPECIAL[0]:
    pp = get_push_pull(contact, freq, months, rebuild)
    if pp.empty or len(pp) < 4:
        st.info("Not enough back-and-forth to model push–pull.")
    else:
        # PRIMARY: signed push–pull index (polarization) with smoothed trend + zones.
        pol = pp["polarization"]
        win = max(3, len(pol) // 8)
        smooth = pol.rolling(win, center=True, min_periods=1).mean()
        fig = go.Figure()
        fig.add_hrect(y0=0, y1=pol.max() * 1.2 + 0.1, fillcolor="rgba(76,139,245,0.07)", line_width=0)
        fig.add_hrect(y0=pol.min() * 1.2 - 0.1, y1=0, fillcolor="rgba(232,97,140,0.07)", line_width=0)
        fig.add_trace(go.Scatter(x=pol.index, y=pol.values, mode="markers",
                                 marker=dict(size=5, color="#888", opacity=0.45), name="per period"))
        fig.add_trace(go.Scatter(x=smooth.index, y=smooth.values, mode="lines",
                                 line=dict(width=4, color="#9467bd"), name=f"trend ({win}-pd avg)"))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_annotation(xref="paper", x=0.01, y=0.97, yref="paper", showarrow=False,
                           text=f"▲ {me_name} pushes · {them_name} pulls back", font=dict(color=COLOR_ME, size=12))
        fig.add_annotation(xref="paper", x=0.01, y=0.03, yref="paper", showarrow=False,
                           text=f"▼ {them_name} pushes · {me_name} pulls back", font=dict(color=COLOR_THEM, size=12))
        _chart(fig, "Push–pull index over time", height=420, legend=True)
        fig.update_layout(yaxis_title="← they push    ·    you push →")
        st.plotly_chart(fig, use_container_width=True)
        recent = smooth.dropna().iloc[-1] if smooth.notna().any() else 0
        who_push = (f"**you** tend to pursue while **{them_name}** pulls back"
                    if recent > 0.05 else
                    f"**{them_name}** tends to pursue while **you** pull back"
                    if recent < -0.05 else "you're **fairly balanced**")
        st.caption(
            f"The signed index = (your lean-in − pull-back) − ({them_name}'s lean-in − pull-back). "
            f"Above 0 = you push/they withdraw; below = they push/you withdraw; near 0 = symmetric. "
            f"Most recently, {who_push}. The thick line is the smoothed trend so you can read the **evolution**."
        )

        # SECONDARY: coupling — how locked-together the cycle is (rolling correlation).
        cwin = max(4, len(pp) // 5)
        coupling = pp["me_pursuit"].rolling(cwin).corr(pp["them_withdrawal"])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=coupling.index, y=coupling.values, mode="lines",
                                  line=dict(width=3, color="#2ca02c")))
        fig2.add_hline(y=0, line_dash="dot", line_color="gray")
        _chart(fig2, f"Push–pull coupling (rolling {cwin}-pd correlation)", height=300)
        fig2.update_layout(yaxis=dict(range=[-1, 1], title="corr(your pursuit, their withdrawal)"))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "**Coupling** = how tightly your pursuit and their withdrawal move together. "
            "High positive stretches = an active push–pull cycle (you reach out → they pull back, "
            "in lockstep). Near zero = the two aren't reacting to each other."
        )

# ---- Gottman ----
elif section == SPECIAL[1]:
    gs = get_gottman_summary(contact, months, rebuild)
    g = get_gottman(contact, freq, months, rebuild)
    pr_ratio = gs.get("positivity_ratio", 0)
    m1, m2, m3 = st.columns(3)
    m1.metric("Positivity ratio", f"{pr_ratio:.1f}:1",
              "healthy ≥5:1" if pr_ratio >= 5 else "below 5:1 target")
    m2.metric("Harsh startups", f"{gs.get('harsh_startup_pct', 0):.0f}%",
              "of conversation openers are negative", delta_color="off")
    m3.metric("Repair attempts", f"{gs.get('repair_per_100', 0):.1f}", "per 100 msgs", delta_color="off")

    # Positivity ratio over time.
    if "positivity_ratio" in g["shared"]:
        pr = g["shared"]["positivity_ratio"].dropna()
        fig = go.Figure()
        _add_scatter_with_fit(fig, pr.index, pr.values, "Positivity ratio", "#2ca02c")
        fig.add_hline(y=5, line_dash="dash", line_color="green",
                      annotation_text="5:1 healthy", annotation_position="top left")
        fig.add_hline(y=1, line_dash="dot", line_color="red",
                      annotation_text="1:1 distress", annotation_position="bottom left")
        _chart(fig, "Positivity ratio (Gottman 5:1)", height=320)
        st.plotly_chart(fig, use_container_width=True)

    me_g, them_g = g["me"], g["them"]

    def _windowed_rate(gdf, cols=None):
        """Rolling per-100-message rate (volume-normalized, sparse bins trimmed)."""
        if gdf.empty:
            return pd.Series(dtype=float)
        cols = cols or gottman.HORSEMEN
        num = gdf[cols].sum(axis=1) if isinstance(cols, list) else gdf[cols]
        return _safe_rate(num, gdf["volume"])

    # 1) COMBINED conflict trend — the "are we better or worse?" view.
    st.markdown("#### Are conflict signals rising or falling?")
    rate_me, rate_them = _windowed_rate(me_g), _windowed_rate(them_g)
    fig = go.Figure()
    if not rate_me.empty:
        fig.add_trace(go.Scatter(x=rate_me.index, y=rate_me.values, mode="lines",
                                 line=dict(width=3, color=COLOR_ME), name=me_name))
    if not rate_them.empty:
        fig.add_trace(go.Scatter(x=rate_them.index, y=rate_them.values, mode="lines",
                                 line=dict(width=3, color=COLOR_THEM), name=them_name))
    _chart(fig, "Four Horsemen — smoothed rate (per 100 messages)", height=340, legend=True)
    fig.update_layout(yaxis_title="conflict markers / 100 msgs")
    st.plotly_chart(fig, use_container_width=True)
    # Verdict: compare first third vs last third of the combined rate.
    both = pd.concat([rate_me, rate_them], axis=1).mean(axis=1).dropna()
    if len(both) >= 4:
        third = max(1, len(both) // 3)
        early, late = both.iloc[:third].mean(), both.iloc[-third:].mean()
        if late < early * 0.85:
            v = f"📉 **Improving** — conflict markers fell from ~{early:.1f} to ~{late:.1f} per 100 msgs."
        elif late > early * 1.15:
            v = f"📈 **Worsening** — conflict markers rose from ~{early:.1f} to ~{late:.1f} per 100 msgs."
        else:
            v = f"➡️ **Stable** — roughly steady (~{late:.1f} per 100 msgs)."
        st.info(v)

    # 2) PER-HORSEMAN RATE — relative change (volume-normalized), can rise OR fall.
    st.markdown("#### Each Horseman — rate per 100 messages (rising = worse, falling = better)")
    st.caption("Normalized by message volume, so a busy month doesn't look worse just "
               "because there were more texts. This is the **relative-change** view.")
    def _combined_rate(gdf_a, gdf_b, col):
        cnt = gdf_a[col].add(gdf_b[col], fill_value=0) if col in gdf_a and col in gdf_b else pd.Series(dtype=float)
        vol = gdf_a["volume"].add(gdf_b["volume"], fill_value=0)
        return _safe_rate(cnt, vol)

    hcols = st.columns(2)
    for i, key in enumerate(gottman.HORSEMEN):
        fig = go.Figure()
        _person_series(fig, _windowed_rate(me_g, key), _windowed_rate(them_g, key),
                       combined_s=_combined_rate(me_g, them_g, key), scatter=False)
        _chart(fig, gottman.LABELS[key], legend=True)
        fig.update_layout(yaxis_title="per 100 msgs")
        with hcols[i % 2]:
            st.plotly_chart(fig, use_container_width=True)

    # 3) INSPECT — every flagged conflict message, so odd points are readable.
    st.markdown("#### 🔍 Inspect the actual conflict messages")
    ge = gottman.enrich_gottman(sub)
    ge["type"] = ge[gottman.HORSEMEN].idxmax(axis=1).where(ge[gottman.HORSEMEN].sum(axis=1) > 0)
    flagged = ge[ge["type"].notna()].copy()
    flagged["who"] = flagged["is_from_me"].map({True: me_name, False: them_name})
    pick = st.multiselect("Show types", gottman.HORSEMEN, default=["contempt", "criticism"])
    show = flagged[flagged["type"].isin(pick)][["dt", "who", "type", "text"]].rename(columns={"dt": "when"})
    st.caption(f"{len(show)} flagged messages — read them to judge if the detector got it right.")
    st.dataframe(show.sort_values("when"), use_container_width=True, height=340, hide_index=True)

    st.caption(
        "**Text proxies, not clinical diagnoses.** Four Horsemen (Gottman): criticism, "
        "contempt, defensiveness, stonewalling predict breakdown; contempt is the strongest. "
        "Smoothed rate shows the trend; cumulative shows timing; the table lets you verify each hit."
    )

# ---- Who leads ----
elif section == SPECIAL[2]:
    wl = get_who_leads(contact, freq, months, rebuild)
    if not wl:
        st.info("Not enough data.")
    else:
        st.markdown(
            "**Who sets the emotional tone?** When one person sends an emotional "
            "message, how often does the other **mirror it** in their next reply? "
            "The person whose emotion gets mirrored *more* is the **leader**; the "
            "other **follows**."
        )

        def _leader_block(col, kind, mine_caught, theirs_caught, color):
            # mine_caught = how often THEY mirror MINE  -> if high, I lead.
            # theirs_caught = how often I mirror THEIRS -> if high, they lead.
            if abs(mine_caught - theirs_caught) < 0.03:
                leader, follower, hi, lo = "Mutual", "", mine_caught, theirs_caught
            elif mine_caught > theirs_caught:
                leader, follower, hi, lo = me_name, them_name, mine_caught, theirs_caught
            else:
                leader, follower, hi, lo = them_name, me_name, theirs_caught, mine_caught
            with col:
                st.markdown(f"#### {kind}")
                if leader == "Mutual":
                    st.metric("Leader", "Mutual", "roughly even", delta_color="off")
                else:
                    st.metric("Leader 👑", leader,
                              f"{follower} mirrors {leader} {hi:.0%} of the time", delta_color="off")
                # Highlighted comparison bar: which reciprocation is bigger.
                fig = go.Figure(go.Bar(
                    x=[f"{them_name} mirrors\n{me_name}'s {kind.lower()}",
                       f"{me_name} mirrors\n{them_name}'s {kind.lower()}"],
                    y=[100 * mine_caught, 100 * theirs_caught],
                    marker_color=[color if mine_caught >= theirs_caught else "#555",
                                  color if theirs_caught > mine_caught else "#555"],
                    text=[f"{mine_caught:.0%}", f"{theirs_caught:.0%}"], textposition="outside"))
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                                  yaxis_title="% mirrored", showlegend=False,
                                  yaxis=dict(range=[0, max(mine_caught, theirs_caught) * 130 + 5]))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"The **taller, colored bar wins** — that emotion is more contagious, "
                           f"so its sender (**{leader if leader!='Mutual' else 'neither clearly'}**) leads {kind.lower()}.")

        cols = st.columns(2)
        _leader_block(cols[0], "Negativity", wl["neg_me_to_them"], wl["neg_them_to_me"], "#E74C3C")
        _leader_block(cols[1], "Positivity", wl["pos_me_to_them"], wl["pos_them_to_me"], "#2ca02c")

        st.divider()
        st.markdown("**Lead–lag mood correlation** (corroborating view)")
        cc = wl.get("crosscorr", {})
        if cc:
            lags = sorted(cc)
            peak = max(cc, key=lambda k: cc[k])
            colors = ["#9467bd" if l != peak else "#FFA500" for l in lags]
            fig2 = go.Figure(go.Bar(x=[str(l) for l in lags], y=[cc[l] for l in lags],
                                    marker_color=colors))
            fig2.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                               xaxis_title=f"lag: ◀ {me_name} leads    ·    {them_name} leads ▶",
                               yaxis_title="sentiment correlation",
                               title=dict(text="Highlighted bar = strongest alignment", font=dict(size=14)))
            st.plotly_chart(fig2, use_container_width=True)
            who = me_name if peak < 0 else them_name if peak > 0 else "neither (synchronous)"
            st.caption(f"Correlation peaks at lag **{peak}** (orange) → **{who} tends to lead** the overall "
                       "mood: their sentiment in one period predicts the other's in the next.")

# ---- Trust signals ----
elif section == SPECIAL[3]:
    st.warning(
        "**How to read this — please.** Text can show conversational *patterns related to* "
        "trust (promises, accusations, accountability, follow-through language). It **cannot** "
        "measure whether someone is actually trustworthy, and it **cannot** tell you whether "
        f"{them_name}'s feelings are right or wrong — feelings aren't validated or refuted by "
        "message stats. Use this to **reflect and find things worth talking about**, not as a "
        "verdict. Trust is rebuilt through real-world follow-through and honest conversation "
        "(often with a therapist), not a dashboard."
    )
    t = get_trust(contact, freq, months, rebuild)
    tsum = trust.summary(sub)
    me_t, them_t = t["me"], t["them"]

    def _trate(gdf, col):
        if gdf.empty:
            return pd.Series(dtype=float)
        return _safe_rate(gdf[col], gdf["volume"])

    # Headline cards most relevant to "she doesn't trust me".
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{them_name}: distrust voiced", f"{tsum.get('distrust_them', 0):.1f}", "per 100 msgs", delta_color="off")
    m2.metric(f"{me_name}: commitments made", f"{tsum.get('commitments_me', 0):.1f}", "per 100 msgs", delta_color="off")
    m3.metric(f"{me_name}: accountability", f"{tsum.get('accountability_me', 0):.1f}", "per 100 msgs", delta_color="off")
    m4.metric(f"{them_name}: trust affirmed", f"{tsum.get('affirmation_them', 0):.1f}", "per 100 msgs", delta_color="off")

    def _combined_trate(col):
        cnt = me_t[col].add(them_t[col], fill_value=0) if col in me_t and col in them_t else pd.Series(dtype=float)
        vol = me_t["volume"].add(them_t["volume"], fill_value=0)
        return _safe_rate(cnt, vol)

    st.markdown("#### Trust-related language over time (rate per 100 messages)")
    tcols = st.columns(2)
    for i, cat in enumerate(trust.CATEGORIES):
        fig = go.Figure()
        _person_series(fig, _trate(me_t, cat), _trate(them_t, cat),
                       combined_s=_combined_trate(cat), scatter=False)
        _chart(fig, trust.LABELS[cat], legend=True)
        fig.update_layout(yaxis_title="per 100 msgs")
        with tcols[i % 2]:
            st.plotly_chart(fig, use_container_width=True)

    # Measured interpretation — patterns, not a verdict.
    d_them = tsum.get("distrust_them", 0)
    commit_me, acct_me = tsum.get("commitments_me", 0), tsum.get("accountability_me", 0)
    notes = []
    if d_them > 0:
        notes.append(f"- **{them_name} voices distrust ~{d_them:.1f}×/100 msgs** — read the actual "
                     "messages below to see *what* the concern is about (broken plans? a recurring issue?).")
    if commit_me > 0:
        notes.append(f"- **{me_name} makes commitments ~{commit_me:.1f}×/100 msgs.** Text can't verify "
                     "follow-through, but the inspector lets you trace a promise to what happened next.")
    notes.append(f"- **{me_name}'s accountability/repair: ~{acct_me:.1f}×/100.** Owning mistakes is one of "
                 "the strongest trust-repair behaviors (Gottman) — worth noticing if it rises after conflict.")
    st.markdown("#### What the patterns suggest")
    st.markdown("\n".join(notes))

    # Inspect — read the actual trust-relevant messages.
    st.markdown("#### 🔍 Read the actual messages")
    fl = trust.flagged_messages(sub)
    fl["who"] = fl["is_from_me"].map({True: me_name, False: them_name})
    pick = st.multiselect("Categories", trust.CATEGORIES, default=["distrust", "commitments"])
    show = fl[fl["category"].isin(pick)][["dt", "who", "category", "text"]].rename(columns={"dt": "when"})
    st.caption(f"{len(show)} messages. Reading these in context is far more honest than any single number.")
    st.dataframe(show.sort_values("when"), use_container_width=True, height=360, hide_index=True)

# ---- Personality (MBTI-style) ----
elif section == SPECIAL[4]:
    p = get_personality(contact, months, rebuild)
    st.caption("Type indicators estimated from writing style across the four MBTI "
               "dichotomies. A complementary read on language patterns — most reliable "
               "with a few hundred+ messages per person.")

    def _render_type(col, label, r, color):
        with col:
            if not r:
                st.info(f"Not enough messages for {label}.")
                return
            MARGIN = 6  # within ±6% of 50 = essentially even, don't over-read
            typ = r["type"]
            # Lowercase letters whose axis is a near-tie, to signal low confidence.
            pcts = [r[typ[0]], r[typ[1]], r[typ[2]], r[typ[3]]]
            disp = "".join(c if abs(p - 50) >= MARGIN else c.lower()
                           for c, p in zip(typ, pcts))
            st.markdown(f"#### {label}")
            st.markdown(f"<div style='font-size:48px;font-weight:700;color:{color};"
                        f"line-height:1'>{disp}</div>", unsafe_allow_html=True)
            st.caption(personality.TYPE_DESC.get(typ, ""))
            for a, b in personality.AXES:
                hi, lo = (a, b) if r[a] >= r[b] else (b, a)
                fn = personality.FULL_NAME
                even = abs(r[hi] - 50) < MARGIN
                note = "  &nbsp;·&nbsp; ⚖️ *too close to call*" if even else ""
                st.markdown(f"**{fn[hi]} {r[hi]:.0f}%** &nbsp;·&nbsp; {fn[lo]} {r[lo]:.0f}%{note}",
                            unsafe_allow_html=True)
                st.progress(int(r[hi]))
            st.caption(f"based on {r['n']:,} messages · lowercase letters above = "
                       "an essentially even axis (don't read a real difference there)")

    c1, c2 = st.columns(2)
    _render_type(c1, me_name, p.get("me", {}), COLOR_ME)
    _render_type(c2, them_name, p.get("them", {}), COLOR_THEM)

    st.divider()
    st.caption("Dichotomies: **E/I** extraversion · **N/S** intuition/sensing · "
               "**F/T** feeling/thinking · **P/J** perceiving/judging. Estimated from "
               "word-usage proxies (social/abstract/emotion/planning language), not a "
               "formal questionnaire — use as a complementary indicator.")
