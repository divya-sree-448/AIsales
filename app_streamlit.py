# app_streamlit.py
import time
import threading
import numpy as np
import streamlit as st
import re
from datetime import datetime

from speech_to_text import calibrate_silence, record_until_silence
from sentiment import analyze_audio
from google_sheets import ensure_headers, save_to_sheets
from config import SAMPLE_RATE, CHANNELS, SILENCE_LIMIT, sheet

# ---------------- Page Setup ----------------
st.set_page_config(page_title="AI Speech Analysis Studio", page_icon="🎙️", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.app-title { font-size: 36px; font-weight: 800; color: #111827; margin-bottom: 4px;}
.app-subtitle { color: #6B7280; margin-bottom: 20px; }
.navbar { display: inline-flex; gap: 8px; padding: 6px; background: #fff; border-radius: 14px; 
          box-shadow: 0 6px 16px rgba(17,24,39,.08); margin-top: 8px; margin-bottom: 18px; }
.navbtn { padding: 8px 14px; border-radius: 10px; border: 1px solid #E5E7EB; background: #fff; 
          color: #111827; font-weight: 600; }
.navbtn.active { background: #111827; color: #fff; border-color: #111827; }
.card { background: #fff; border: 1px solid #E5E7EB; border-radius: 16px; padding: 18px; 
        box-shadow: 0 8px 20px rgba(17,24,39,.05); }
.badge { display: inline-block; padding: 6px 10px; border-radius: 999px; font-weight: 700; font-size: 13px; }
.badge.pos { color:#065F46; background:#D1FAE5; }
.badge.neg { color:#991B1B; background:#FEE2E2; }
.badge.neu { color:#374151; background:#E5E7EB; }
.badge.emo { color:#111827; background:#EDE9FE; }
.transcript { border: 1px solid #E5E7EB; background: #F9FAFB; border-radius: 12px; padding: 12px; 
              min-height: 140px; color:#111827; }
.small { color:#6B7280; font-size: 13px; }
.placeholder { color:#9CA3AF; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown('<div class="app-title">🎙️ AI Speech Analysis Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Real-time Speech-to-Text with Sentiment & Emotion Analysis</div>', unsafe_allow_html=True)

# ---------------- Tabs ----------------
tab = st.session_state.get("tab", "Record")
c1, c2, c3 = st.columns([1,1,1])
with c1:
    if st.button("Record", key="nav_record"): tab = "Record"
with c2:
    if st.button("History", key="nav_history"): tab = "History"
with c3:
    if st.button("Analytics", key="nav_analytics"): tab = "Analytics"
st.session_state["tab"] = tab

st.markdown(
    f"""
    <div class="navbar">
      <span class="{"navbtn active" if tab=="Record" else "navbtn"}">Record</span>
      <span class="{"navbtn active" if tab=="History" else "navbtn"}">History</span>
      <span class="{"navbtn active" if tab=="Analytics" else "navbtn"}">Analytics</span>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- helpers ----------------
def _background_capture(threshold, holder, stop_event):
    audio_list, stop_reason = record_until_silence(threshold, stop_event=stop_event)
    holder["audio_list"] = audio_list
    holder["stop_reason"] = stop_reason
    holder["done"] = True

def _canonical_or_seen(label: str, domain: str) -> str:
    """
    Keep canonical labels when possible, otherwise return a cleaned version
    of whatever the model produced (e.g., 'Observation', 'Calm', etc.).
    domain: 'sent' or 'emo'
    """
    if not label:
        return ""
    s = str(label).strip()
    # normalize: grab first token, strip punctuation -> Title case
    first = re.sub(r"[^\w\-]+", " ", s).split()[0].title() if s else ""

    if domain == "sent":
        if first in {"Positive", "Negative", "Neutral"}:
            return first
        # try to find canonical name anywhere in the string
        for c in ["Positive", "Negative", "Neutral"]:
            if c.lower() in s.lower():
                return c
    else:
        if first in {"Joy", "Sadness", "Anger", "Fear", "Surprise"}:
            return first
        for c in ["Joy", "Sadness", "Anger", "Fear", "Surprise"]:
            if c.lower() in s.lower():
                return c

    # fallback to cleaned token (so bars appear for unknowns like 'Observation')
    return first or s.title()
def refresh_animation(flag_key="_do_refresh"):
    """
    If the refresh flag is set in session_state, show a short animation
    and then clear the flag and rerun.
    """
    if st.session_state.get(flag_key):
        with st.status("Refreshing data…", expanded=False) as s:
            # quick animated effect
            ph = st.empty()
            for dots in ["", ".", "..", "...", "....", "....."]:
                ph.write(f"Updating{dots}")
                time.sleep(0.2)
            s.update(label="Data refreshed ✅")
            time.sleep(0.3)
        st.session_state[flag_key] = False
        st.rerun()


# ---------------- RECORD ----------------
if tab == "Record":
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Voice Recorder")
        st.caption("Toggle Start/Stop. Stops automatically on silence, too.")
        st.write(f"**Sample Rate:** {SAMPLE_RATE/1000:.0f} kHz | **Channels:** {'Mono' if CHANNELS==1 else CHANNELS} | **Silence Limit:** {SILENCE_LIMIT}s")
        st.divider()

        # init state
        st.session_state.setdefault("rec_thread", None)
        st.session_state.setdefault("rec_holder", {})
        st.session_state.setdefault("rec_stop", None)
        st.session_state.setdefault("rec_start_ts", None)
        st.session_state.setdefault("is_recording", False)

        # toggle button
        label = "⏹️ Stop Recording" if st.session_state.is_recording else "🔴 Start Recording"
        toggled = st.button(label, use_container_width=True)

        if toggled:
            if not st.session_state.is_recording:
                # clear old results
                for k in ("audio", "transcript", "sentiment", "emotion", "stop_reason", "timestamp"):
                    if k in st.session_state: del st.session_state[k]

                with st.status("Calibrating baseline noise…", expanded=True) as s:
                    thr = calibrate_silence()
                    s.write(f"Calibrated threshold = {thr:.6f}")
                    s.update(label="Listening… Speak now.")

                holder = {"done": False}
                stop_event = threading.Event()
                t = threading.Thread(target=_background_capture, args=(thr, holder, stop_event), daemon=True)
                t.start()

                st.session_state.rec_holder = holder
                st.session_state.rec_stop = stop_event
                st.session_state.rec_thread = t
                st.session_state.rec_start_ts = time.time()
                st.session_state.is_recording = True
            else:
                if st.session_state.rec_stop is not None:
                    st.session_state.rec_stop.set()

        # timer while recording
        if st.session_state.is_recording and st.session_state.rec_thread and st.session_state.rec_thread.is_alive():
            elapsed = int(time.time() - (st.session_state.rec_start_ts or time.time()))
            st.markdown(f"**⏱️ Recording:** {elapsed:02d} sec")
            time.sleep(1)
            st.rerun()

        # after recording stops
        if st.session_state.rec_thread is not None and not st.session_state.rec_thread.is_alive():
            holder = st.session_state.rec_holder or {}
            if holder.get("done") and "audio" not in st.session_state:
                audio_list = holder.get("audio_list") or []
                stop_reason = holder.get("stop_reason", "")
                st.session_state["stop_reason"] = stop_reason

                # if silence stop, trim the tail to avoid dot-only transcripts (optional)
                if stop_reason.lower().startswith("silent") and len(audio_list) > SILENCE_LIMIT:
                    audio_list = audio_list[:len(audio_list)-SILENCE_LIMIT]

                if audio_list:
                    merged = np.concatenate(audio_list, axis=0)
                    st.session_state["audio"] = merged
                    st.session_state["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    st.toast(f"Captured {merged.shape[0]/SAMPLE_RATE:.1f} sec", icon="🎧")
                else:
                    st.warning("No audio captured.")

            # reset recording state
            st.session_state.rec_thread = None
            st.session_state.rec_stop = None
            st.session_state.rec_holder = {}
            st.session_state.rec_start_ts = None
            st.session_state.is_recording = False
            st.rerun()

        # auto-analyze
        if "audio" in st.session_state and "transcript" not in st.session_state:
            with st.spinner("Analyzing…"):
                transcript, sentiment_label, emotion_label = analyze_audio(
                    st.session_state["audio"],
                    st.session_state.get("stop_reason","")
                )
            st.session_state["transcript"] = transcript
            st.session_state["sentiment"] = sentiment_label
            st.session_state["emotion"] = emotion_label

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Analysis Results")
        st.caption("Transcription and sentiment analysis")

        st.markdown("**Transcript**")
        transcript_text = st.session_state.get("transcript", None)
        if transcript_text is None:
            st.markdown('<div class="transcript"><span class="placeholder">Waiting for new recording...</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="transcript">{transcript_text}</div>', unsafe_allow_html=True)

        st.markdown(" ")
        a, b = st.columns(2)
        with a:
            st.markdown("**Sentiment**")
            sent = st.session_state.get("sentiment", "—")
            css = "neu"
            if isinstance(sent, str):
                s = sent.strip().lower()
                if "pos" in s: css = "pos"
                elif "neg" in s: css = "neg"
            st.markdown(f'<span class="badge {css}">{sent}</span>', unsafe_allow_html=True)
        with b:
            st.markdown("**Emotion**")
            emo = st.session_state.get("emotion", "—")
            st.markdown(f'<span class="badge emo">{emo}</span>', unsafe_allow_html=True)

        st.markdown(" ")
        if st.button("💾 Save to Google Sheets", use_container_width=True):
            ts = st.session_state.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
            try:
                ensure_headers()
                save_to_sheets(
                    ts,
                    st.session_state.get("transcript",""),
                    st.session_state.get("sentiment",""),
                    st.session_state.get("emotion",""),
                    st.session_state.get("stop_reason","")
                )
                st.success("Saved to Google Sheets.")
            except Exception as e:
                st.error(f"Save failed: {e}")

        stop_reason = st.session_state.get("stop_reason", "")
        if stop_reason:
            st.markdown(f'<div class="small">Stop Reason: <b>{stop_reason}</b></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- HISTORY ----------------
if tab == "History":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Recording History")
    st.caption("Newest first")

    try:
        values = sheet.get_all_values()
        if values and len(values) > 1:
            headers = values[0]
            rows = values[1:]

            # sort by Timestamp column (desc). Fallback safely if missing/bad.
            ts_idx = headers.index("Timestamp") if "Timestamp" in headers else 0

            def parse_ts(r):
                try:
                    return datetime.strptime(r[ts_idx], "%Y-%m-%d %H:%M:%S")
                except Exception:
                    return datetime.min

            rows = sorted(rows, key=parse_ts, reverse=True)

            table = [
                {headers[i]: (r[i] if i < len(r) else "") for i in range(len(headers))}
                for r in rows
            ]
            st.dataframe(table, use_container_width=True, height=480)
        else:
            st.info("No rows yet.")
    except Exception as e:
        st.error(f"Could not read Google Sheets: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- ANALYTICS ----------------
if tab == "Analytics":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Analytics")
    st.caption("Distributions from your saved history")

    # Optional manual refresh (with animation)
    col_refresh, _ = st.columns([1, 6])
    with col_refresh:
        if st.button("↻ Refresh data", use_container_width=False):
            st.session_state["_do_refresh"] = True
            st.rerun()

    # Play the animation if the flag was set
    refresh_animation("_do_refresh")

    try:
        values = sheet.get_all_values()
        headers = values[0] if values else []
        rows = values[1:] if values and len(values) > 1 else []

        def col_idx(name):
            try:
                return headers.index(name)
            except ValueError:
                return None

        i_sent = col_idx("Sentiment")
        i_emo  = col_idx("Emotion")

        total = len(rows)
        st.metric("Total Recordings", total)

        # Count everything (including non-canon like 'Observation')
        sent_counts, emo_counts = {}, {}
        for r in rows:
            if i_sent is not None and i_sent < len(r):
                lbl = _canonical_or_seen(r[i_sent], "sent")
                if lbl:
                    sent_counts[lbl] = sent_counts.get(lbl, 0) + 1
            if i_emo is not None and i_emo < len(r):
                lbl = _canonical_or_seen(r[i_emo], "emo")
                if lbl:
                    emo_counts[lbl] = emo_counts.get(lbl, 0) + 1

        c1, c2 = st.columns(2)

        # --- Sentiment Distribution ---
        with c1:
            st.markdown("**Sentiment Distribution**")
            if sent_counts:
                canon_sent = ["Positive", "Neutral", "Negative"]
                extras = sorted([k for k in sent_counts.keys() if k not in canon_sent])
                order = canon_sent + extras
                data_abs = {k: sent_counts.get(k, 0) for k in order if k in sent_counts}
                st.bar_chart(data_abs, height=260)
            else:
                st.info("No data yet.")

        # --- Emotion Distribution ---
        with c2:
            st.markdown("**Emotion Distribution**")
            if emo_counts:
                canon_emo = ["Joy", "Sadness", "Anger", "Fear", "Surprise"]
                extras = sorted([k for k in emo_counts.keys() if k not in canon_emo])
                order = canon_emo + extras
                data_abs = {k: emo_counts.get(k, 0) for k in order if k in emo_counts}
                st.bar_chart(data_abs, height=260)
            else:
                st.info("No data yet.")
    except Exception as e:
        st.error(f"Analytics error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

