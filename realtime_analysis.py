import os
import sounddevice as sd
import numpy as np
import tempfile
import wave
import datetime
import time
import traceback
from groq import Groq
import gspread
from google.oauth2.service_account import Credentials

# -------------------------
# Configuration / Settings
# -------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # set this in your environment
if not GROQ_API_KEY:
    print("⚠ Warning: GROQ_API_KEY not set. Set it in your environment before transcription.")

# Groq client
groq_client = Groq(api_key=GROQ_API_KEY)


# Recording
SAMPLE_RATE = 16000
CHANNELS = 1
SESSION_DURATION = 180   # seconds
SILENCE_LIMIT = 5        # number of consecutive 1-second chunks considered silence

# Google Sheets
SPREADSHEET_KEY = "12BSvCzKWccU8DlhANNbW-94p1D73rH4Lkjojhg-URfo"
SERVICE_ACCOUNT_FILE = "credentials.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# -------------------------
# Helper utilities
# -------------------------
def check_service_account_file():
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise FileNotFoundError(
            f"Service account JSON not found: '{SERVICE_ACCOUNT_FILE}'.\n"
            "Download from Google Cloud Console and place it in the script folder, then re-run."
        )

def safe_get_choice_content(choice):
    """ Extract response content safely from Groq API response """
    try:
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")

        if isinstance(message, dict):
            return message.get("content") or message.get("text") or ""
        else:
            return getattr(message, "content", None) or getattr(message, "text", None) or ""
    except Exception:
        return ""

def append_row_safe(ws, row):
    try:
        ws.append_row(row, value_input_option='USER_ENTERED')
    except Exception as e:
        print("❌ Failed to append to Google Sheets:", e)

# -------------------------
# Main
# -------------------------
def main():
    try:
        # 1) Google creds + sheet open
        check_service_account_file()
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        gs_client = gspread.authorize(creds)

        try:
            sheet = gs_client.open_by_key(SPREADSHEET_KEY).sheet1
        except Exception as e:
            raise RuntimeError(
                f"Unable to open spreadsheet with key '{SPREADSHEET_KEY}'. "
                "Check the key, ensure the sheet is shared with the service account email, "
                "and that the Drive API is enabled."
            ) from e

        # Ensure headers exist
        if sheet.row_count == 0 or not sheet.row_values(1):
            sheet.append_row(["Timestamp", "Transcript", "Sentiment", "Emotion", "StopReason"])

        # 2) Calibration
        print("Assistant started (max 3 mins, stops if silence >5s)")
        print("\nCalibrating... stay quiet for 3s...")
        calib = sd.rec(int(3 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
        sd.wait()
        baseline = np.linalg.norm(calib) / len(calib)

        SILENCE_THRESHOLD = max(baseline * 1.2, 0.00005)
        print(f"✅ Calibration done. Baseline={baseline:.6f}, Threshold={SILENCE_THRESHOLD:.6f}")

        # 3) Recording loop
        recorded_audio = []
        silence_counter = 0
        start_time = time.time()
        stop_reason = "3min limit reached"

        try:
            while (time.time() - start_time) < SESSION_DURATION:
                chunk = sd.rec(int(SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
                sd.wait()
                recorded_audio.append(chunk)

                elapsed = int(time.time() - start_time)
                volume = np.linalg.norm(chunk) / len(chunk)

                print(f"⏱ {elapsed}s | 🎚 Volume={volume:.6f}", end="\r")

                if volume < SILENCE_THRESHOLD:
                    silence_counter += 1
                    if silence_counter >= SILENCE_LIMIT:
                        stop_reason = f"Silent >{SILENCE_LIMIT}s"
                        break
                else:
                    silence_counter = 0

        except KeyboardInterrupt:
            stop_reason = "Stopped by user"

        print(f"\nRecording stopped: {stop_reason}")

        # 4) Ensure we have audio
        if not recorded_audio:
            print("No audio recorded. Exiting.")
            return

        recording = np.concatenate(recorded_audio, axis=0)

        # 5) Save temp wav
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_filename = tmp_file.name
            with wave.open(tmp_filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)   # 16-bit
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes((recording * 32767).astype(np.int16).tobytes())

        # 6) Transcribe & analyze
        if stop_reason.startswith("Silent"):
            text = "Not Speaking"
            sentiment_result = "N/A"
            emotion_result = "N/A"
        else:
            try:
                with open(tmp_filename, "rb") as f:
                    transcription = groq_client.audio.transcriptions.create(model="whisper-large-v3", file=f)
                text = getattr(transcription, "text", None) or ""
            except Exception as e:
                print("❌ Transcription failed:", e)
                traceback.print_exc()
                text = "[TRANSCRIPTION ERROR]"
                sentiment_result = "N/A"
                emotion_result = "N/A"

            if text and text not in ("", "[TRANSCRIPTION ERROR]"):
                # Sentiment
                try:
                    sentiment = groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role":"system","content":"Reply with only one word: Positive, Negative, or Neutral."},
                            {"role":"user","content": text}
                        ]
                    )
                    sentiment_result = safe_get_choice_content(sentiment.choices[0]).strip()
                except Exception:
                    sentiment_result = "[SENTIMENT ERROR]"
                    traceback.print_exc()

                # Emotion
                try:
                    emotion = groq_client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role":"system","content":"Reply with only one word: Joy, Sadness, Anger, Fear, or Surprise."},
                            {"role":"user","content": text}
                        ]
                    )
                    emotion_result = safe_get_choice_content(emotion.choices[0]).strip()
                except Exception:
                    emotion_result = "[EMOTION ERROR]"
                    traceback.print_exc()
            else:
                sentiment_result = "N/A"
                emotion_result = "N/A"

        # 7) Save to Google Sheets
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        append_row_safe(sheet, [timestamp, text, sentiment_result, emotion_result, stop_reason])

        # 8) Final prints
        print(f"\n📝 Transcript: {text}")
        print(f"📊 Sentiment: {sentiment_result} | 🎭 Emotion: {emotion_result}")
        print("✅ Results saved to Google Sheets")

    except Exception as overall_e:
        print("❌ Fatal error occurred. See trace below:")
        traceback.print_exc()

if __name__ == "__main__":
    main()