



# app_streamlit.py
import time
import threading
import numpy as np
import streamlit as st
import re
import pandas as pd
from datetime import datetime

from speech_to_text import calibrate_silence, record_until_silence
from sentiment import analyze_audio
from google_sheets import ensure_headers, save_to_sheets
from config import SAMPLE_RATE, CHANNELS, SILENCE_LIMIT, sheet, client  # client for LLM summary

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
      <span class="{'navbtn active' if tab=='Record' else 'navbtn'}">Record</span>
      <span class="{'navbtn active' if tab=='History' else 'navbtn'}">History</span>
      <span class="{'navbtn active' if tab=='Analytics' else 'navbtn'}">Analytics</span>
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

def refresh_animation(flag_key="_do_refresh"):
    if st.session_state.get(flag_key):
        with st.status("Refreshing data…", expanded=False) as s:
            ph = st.empty()
            for dots in ["", ".", "..", "...", "....", "....."]:
                ph.write(f"Updating{dots}")
                time.sleep(0.2)
            s.update(label="Data refreshed ✅")
            time.sleep(0.3)
        st.session_state[flag_key] = False
        st.rerun()

def _normalize_label(txt: str, domain: str) -> str:
    if not txt:
        return ""
    s = (txt or "").strip()
    first = s.split()[0].strip(",. ").title()
    if domain == "sent":
        if first in {"Positive", "Negative", "Neutral"}:
            return first
    if domain == "emo":
        if first in {"Joy", "Sadness", "Anger", "Fear", "Surprise"}:
            return first
    candidates = (["Positive","Negative","Neutral"] if domain=="sent"
                 else ["Joy","Sadness","Anger","Fear","Surprise"])
    for c in candidates:
        if c.lower() in s.lower():
            return c
    return ""

def _clear_last_analysis():
    for k in ("audio", "transcript", "sentiment", "emotion", "stop_reason", "timestamp"):
        st.session_state.pop(k, None)
    # keep a placeholder so UI shows "Waiting..."
    st.session_state["transcript"] = None
    st.session_state["sentiment"] = "—"
    st.session_state["emotion"] = "—"

# ==== CRM CONFIG ====
CRM_SHEET_NAME = "CRM"
SUMMARIES_SHEET_NAME = "Summaries"

CRM_HEADERS = [
    "CustomerName", "Company", "Industry",
    "Budget", "InterestLevel", "Email", "Phone","RecommendedProducts"
]

SUMMARIES_HEADERS = [
    "Timestamp", "CustomerName",
    "Summary", "ActionItems", "Sentiment", "Emotion"
]

def _get_ws(title: str):
    ss = sheet.spreadsheet
    try:
        return ss.worksheet(title)
    except Exception:
        ws = ss.add_worksheet(title=title, rows=1000, cols=20)
        return ws

def _ensure_ws_headers(ws, headers):
    values = ws.get_all_values()
    if not values:
        ws.update('A1', [headers])
    else:
        if values[0] != headers:
            ws.update(f"A1:{chr(64+len(headers))}1", [headers])

def ensure_crm_ready():
    ws = _get_ws(CRM_SHEET_NAME)
    _ensure_ws_headers(ws, CRM_HEADERS)
    return ws

def ensure_summaries_ready():
    ws = _get_ws(SUMMARIES_SHEET_NAME)
    _ensure_ws_headers(ws, SUMMARIES_HEADERS)
    return ws

@st.cache_data(ttl=300)
def load_crm_df() -> pd.DataFrame:
    ws = ensure_crm_ready()
    values = ws.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame(columns=CRM_HEADERS)
    df = pd.DataFrame(values[1:], columns=values[0])
    if "Budget" in df.columns:
        df["Budget"] = pd.to_numeric(df["Budget"], errors="coerce")
    return df

# ---------- Email-based helpers ----------
def get_customer_options(df: pd.DataFrame):
    """
    Return (labels, id_map) where id_map maps label -> Email (unique key).
    We show "Name — Company" in the dropdown.
    """
    labels = []
    id_map = {}
    for _, row in df.iterrows():
        name = row.get("CustomerName", "") or ""
        company = row.get("Company", "") or ""
        email = row.get("Email", "") or ""
        label = f"{name} — {company}" if company else name
        labels.append(label)
        id_map[label] = email
    return labels, id_map

def get_customer_by_email(df: pd.DataFrame, email: str) -> dict:
    if not email:
        return {}
    hit = df.loc[df["Email"] == email]
    if hit.empty:
        return {}
    return hit.iloc[0].to_dict()

def parse_products(cell: str):
    if not cell:
        return []
    parts = [p.strip() for p in str(cell).split(",") if p.strip()]
    seen, out = set(), []
    for p in parts:
        low = p.lower()
        if low not in seen:
            out.append(p)
            seen.add(low)
    return out

def rank_products(products, sentiment: str):
    if not isinstance(sentiment, str) or not products:
        return products
    s = sentiment.strip().lower()
    if "neg" in s:
        soft, hard = [], []
        for p in products:
            pl = p.lower()
            if any(k in pl for k in ["trial", "demo", "lite", "basic"]):
                soft.append(p)
            else:
                hard.append(p)
        return soft + hard if soft else products
    return products

def save_summary_row(timestamp: str, customer: dict, summary: str,
                     action_items: str, sentiment: str, emotion: str):
    ws = ensure_summaries_ready()
    row = [
        timestamp,
        customer.get("CustomerName", ""),
        summary or "",
        action_items or "",
        sentiment or "",
        emotion or "",
    ]
    ws.append_row(row)

# ---- Seed CRM with 25 demo customers ----
def seed_crm_with_25():
    ws = ensure_crm_ready()
    ws.clear()  # wipe old data
    headers = [
        "CustomerName","Company","Industry","Budget",
        "InterestLevel","Email","Phone","RecommendedProducts"
    ]
    rows = [
        ["Aisha Patel","NovaTech Systems","Technology","24000","High",
        "aisha.patel@gmail.com","+91-9801001000","CRM Suite, Analytics Dashboard"],
        ["David Rodrigues","GreenMart Retail","Retail","18000","Medium",
        "david.rod@outlook.com","+91-9802002000","POS System, Loyalty App"],
        ["Priya Deshmukh","MediCare Hospitals","Healthcare","30000","High",
        "priya.desh@yahoo.com","+91-9803003000","Telehealth Platform, Patient CRM"],
        ["Liam Sharma","EduSpark Learning","Education","15000","Medium",
        "liam.sharma@zoho.com","+91-9804004000","LMS Platform, Online Classrooms"],
        ["Kavya Menon","AutoWorks Manufacturing","Manufacturing","22000","High",
        "kavya.menon@gmail.com","+91-9805005000","ERP Suite, Predictive Maintenance"],
        ["Rohan Chatterjee","AgroTech Innovations","Agriculture","20000","Medium",
        "rohan.chat@outlook.com","+91-9806006000","IoT Sensors, Yield Prediction AI"],
        ["Emma Kapoor","BrightLine Retail","Retail","17000","Medium",
        "emma.kapoor@yahoo.com","+91-9807007000","POS System, Loyalty App"],
        ["Harish Iyer","Quantum Technologies","Technology","26000","High",
        "harish.iyer@zoho.com","+91-9808008000","CRM Suite, Analytics Dashboard"],
        ["Sophia D’Costa","HealthBridge Clinics","Healthcare","28000","High",
        "sophia.d@gmail.com","+91-9809009000","Telehealth Platform, Patient CRM"],
        ["Aarav Mehra","LearnSphere EdTech","Education","15500","Medium",
        "aarav.mehra@outlook.com","+91-9810001000","LMS Platform, Online Classrooms"],
        ["Zoe Fernandes","EcoMart Retailers","Retail","17500","Low",
        "zoe.fern@yahoo.com","+91-9811001100","POS System, Loyalty App"],
        ["Ethan Thomas","MacroTech Industries","Manufacturing","23000","High",
        "ethan.thomas@zoho.com","+91-9812001200","ERP Suite, Predictive Maintenance"],
        ["Maya Narang","AgriSense Solutions","Agriculture","20500","Medium",
        "maya.narang@gmail.com","+91-9813001300","IoT Sensors, Yield Prediction AI"],
        ["Benjamin Kaur","CloudNova Systems","Technology","24500","High",
        "ben.kaur@outlook.com","+91-9814001400","CRM Suite, Analytics Dashboard"],
        ["Ishita Verma","SmartEdu Hub","Education","16000","Medium",
        "ishita.verma@yahoo.com","+91-9815001500","LMS Platform, Online Classrooms"],
        ["Nikhil Suresh","HealthTrack Labs","Healthcare","29500","High",
        "nikhil.s@zoho.com","+91-9816001600","Telehealth Platform, Patient CRM"],
        ["Chloe Dutta","BrightAgro Farms","Agriculture","21500","Medium",
        "chloe.dutta@gmail.com","+91-9817001700","IoT Sensors, Yield Prediction AI"],
        ["Arjun Pillai","QuantumLine Retail","Retail","18500","Medium",
        "arjun.pillai@outlook.com","+91-9818001800","POS System, Loyalty App"],
        ["Tara George","EduNext Academy","Education","16500","Medium",
        "tara.george@yahoo.com","+91-9819001900","LMS Platform, Online Classrooms"],
        ["Vedant Joshi","MediNova Hospitals","Healthcare","30500","High",
        "vedant.joshi@zoho.com","+91-9820002000","Telehealth Platform, Patient CRM"],
        ["Olivia Nair","AgroSphere Tech","Agriculture","21000","Medium",
        "olivia.nair@gmail.com","+91-9821002100","IoT Sensors, Yield Prediction AI"],
        ["Karan Malhotra","NovaWorks Systems","Technology","25000","High",
        "karan.malhotra@outlook.com","+91-9822002200","CRM Suite, Analytics Dashboard"],
        ["Aditi Rao","GreenLeaf Retailers","Retail","19000","Medium",
        "aditi.rao@yahoo.com","+91-9823002300","POS System, Loyalty App"],
        ["Lucas Fernandes","EduBridge Learning","Education","17000","Medium",
        "lucas.fern@zoho.com","+91-9824002400","LMS Platform, Online Classrooms"],
        ["Sara Khan","AutoLine Manufacturing","Manufacturing","22500","High",
        "sara.khan@gmail.com","+91-9825002500","ERP Suite, Predictive Maintenance"],
    ]

    ws.update('A1', [headers] + rows)
    load_crm_df.clear()  # bust cache

# ---- LLM Summary generator ----
def generate_llm_summary(transcript: str, customer: dict, sentiment: str, emotion: str) -> tuple[str, str]:
    if not isinstance(transcript, str) or not transcript.strip():
        return ("Not Speaking. No summary generated.", "")

    name = customer.get("CustomerName", "") if customer else ""
    industry = customer.get("Industry", "") if customer else ""

    sys = (
        "You are a sales assistant. Write a concise post-call summary and clear action items.\n"
        "- Keep summary <= 120 words.\n"
        "- Use simple bullet points in Action Items (2-4 items).\n"
        "- Avoid guessing unknown details.\n"
        "Return JSON only."
    )
    user = (
        f"Customer: {name}\n"
        f"Industry: {industry}\n"
        f"Sentiment: {sentiment}\n"
        f"Emotion: {emotion}\n"
        f"Transcript:\n{transcript}\n\n"
        "Return JSON with keys 'summary' and 'action_items' (list of strings)."
    )

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user}
            ],
            temperature=0.2,
        )
        content = (resp.choices[0].message.content or "").strip()
        import json, re as _re
        data = {}
        try:
            data = json.loads(content)
        except Exception:
            m = _re.search(r"\{.*\}", content, flags=_re.S)
            if m:
                data = json.loads(m.group(0))
        summary = str(data.get("summary", "")).strip() or "Summary unavailable."
        items = data.get("action_items", [])
        if not isinstance(items, list):
            items = [str(items)]
        ai = "; ".join([str(x) for x in items if str(x).strip()])[:400]
        return (summary, ai)
    except Exception as e:
        return (f"Summary error: {e}", "")

# ---------------- RECORD ----------------
if tab == "Record":
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Voice Recorder")
        st.caption("Toggle Start/Stop. Stops automatically on silence, too.")
        st.write(f"**Sample Rate:** {SAMPLE_RATE/1000:.0f} kHz | **Channels:** {'Mono' if CHANNELS==1 else CHANNELS} | **Silence Limit:** {SILENCE_LIMIT}s")
        st.divider()

        # Seed demo CRM if empty
        crm_df_preview = load_crm_df()
        if crm_df_preview.empty:
            if st.button("🌱 Seed CRM with 25 demo names"):
                try:
                    seed_crm_with_25()
                    st.success("CRM seeded with 25 demo customers.")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to seed CRM: {e}")

        # ==== CRM: Customer picker + Profile (real-time) ====
        crm_df = load_crm_df()
        if crm_df.empty:
            st.info("Add some rows to the **CRM** sheet to enable real-time profile & recommendations.")
        else:
            labels, id_map = get_customer_options(crm_df)
            st.session_state["_id_map_cache"] = id_map
            current_label = st.session_state.get("selected_customer_label")
            selected_label = st.selectbox(
                "Select customer",
                options=["— Select —"] + labels,
                index=(["— Select —"] + labels).index(current_label) if current_label in (["— Select —"] + labels) else 0,
                help="Pick a customer to view profile and recommended products instantly."
            )
            st.session_state["selected_customer_label"] = selected_label

            selected_customer = {}
            if selected_label and selected_label != "— Select —":
                email_key = id_map.get(selected_label, "")
                selected_customer = get_customer_by_email(crm_df, email_key)

            # Profile card
            if selected_customer:
                with st.expander("👤 Customer Profile", expanded=True):
                    colA, colB, colC = st.columns(3)
                    colA.metric("Name", selected_customer.get("CustomerName","—"))
                    colB.metric("Industry", selected_customer.get("Industry","—"))
                    colC.metric("Budget", str(selected_customer.get("Budget","—")))
                    colA.metric("Interest", selected_customer.get("InterestLevel","—"))
                    colB.metric("Email", selected_customer.get("Email","—"))
                    colC.metric("Phone", selected_customer.get("Phone","—"))

                # Recommendations (real-time)
                base_products = parse_products(selected_customer.get("RecommendedProducts",""))
                current_sentiment = st.session_state.get("sentiment", "")
                ranked = rank_products(base_products, current_sentiment)
                with st.expander("🧩 Recommended Products", expanded=True):
                    if ranked:
                        for p in ranked:
                            st.write(f"• {p}")
                    else:
                        st.write("No recommendations on file for this customer.")

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
                # clear old results (your original block)
                for k in ("audio", "transcript", "sentiment", "emotion", "stop_reason", "timestamp"):
                    if k in st.session_state: 
                        del st.session_state[k]
                # ✅ ADD: show placeholder immediately so old transcript is wiped in UI
                st.session_state["transcript"] = None

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
        # ✅ CHANGE: run analysis when transcript is explicitly None (placeholder state)
        if "audio" in st.session_state and st.session_state.get("transcript") is None:
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

        # ---- Save block with LLM Summary to 'Summaries' ----
        st.markdown(" ")
        save_summary_too = st.checkbox("Also save post-call summary to 'Summaries'", value=True)

        if st.button("💾 Save to Google Sheets", use_container_width=True):
            ts = st.session_state.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
            try:
                ensure_headers()
                # Save the main recording row
                save_to_sheets(
                    ts,
                    st.session_state.get("transcript",""),
                    st.session_state.get("sentiment",""),
                    st.session_state.get("emotion",""),
                    st.session_state.get("stop_reason","")
                )
                st.success("Saved to Google Sheets.")

                if save_summary_too:
                    # Resolve selected customer (Email key)
                    selected_label = st.session_state.get("selected_customer_label")
                    selected_customer = {}
                    if selected_label and selected_label != "— Select —":
                        id_map = st.session_state.get("_id_map_cache", {})
                        email_key = id_map.get(selected_label, "")
                        selected_customer = get_customer_by_email(load_crm_df(), email_key) if email_key else {}

                    # Generate LLM summary/action items
                    transcript_val = st.session_state.get("transcript","").strip()
                    sentiment_val = st.session_state.get("sentiment","")
                    emotion_val = st.session_state.get("emotion","")
                    summary, action_items = generate_llm_summary(transcript_val, selected_customer, sentiment_val, emotion_val)

                    try:
                        save_summary_row(ts, selected_customer, summary, action_items, sentiment_val, emotion_val)
                        st.toast("Summary saved to 'Summaries' ✅", icon="📝")
                    except Exception as e:
                        st.warning(f"Summary save skipped: {e}")

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

    col_refresh, _ = st.columns([1, 6])
    with col_refresh:
        if st.button("↻ Refresh data", use_container_width=False):
            st.session_state["_do_refresh"] = True
            st.rerun()
    refresh_animation("_do_refresh")

    try:
        values = sheet.get_all_values()
        headers = values[0] if values else []
        rows = values[1:] if values and len(values) > 1 else []

        def col_idx(name):
            try: return headers.index(name)
            except ValueError: return None

        i_sent = col_idx("Sentiment")
        i_emo  = col_idx("Emotion")

        total = len(rows)
        st.metric("Total Recordings", total)

        sent_counts, emo_counts = {}, {}
        for r in rows:
            if i_sent is not None and i_sent < len(r):
                norm = _normalize_label(r[i_sent], "sent")
                if norm:
                    sent_counts[norm] = sent_counts.get(norm, 0) + 1
            if i_emo is not None and i_emo < len(r):
                norm = _normalize_label(r[i_emo], "emo")
                if norm:
                    emo_counts[norm] = emo_counts.get(norm, 0) + 1

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Sentiment Distribution**")
            if sent_counts:
                data = {k: sent_counts.get(k, 0) for k in ["Positive","Neutral","Negative"] if k in sent_counts}
                st.bar_chart(data, height=260)
            else:
                st.info("No data yet.")
        with c2:
            st.markdown("**Emotion Distribution**")
            if emo_counts:
                data = {k: emo_counts.get(k, 0) for k in ["Joy","Sadness","Anger","Fear","Surprise"] if k in emo_counts}
                st.bar_chart(data, height=260)
            else:
                st.info("No data yet.")
    except Exception as e:
        st.error(f"Analytics error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)