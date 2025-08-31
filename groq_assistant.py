




import os
import sounddevice as sd
import numpy as np
import tempfile
import wave
import csv
import datetime
import time
from groq import Groq
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ✅ Groq setup
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 🎤 Settings
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_LIMIT = 5        # stop if silence ≥5s
csv_file = "groq_transcripts.csv"

# 🔹 Google Sheets setup
scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client_gs = gspread.authorize(creds)
sheet = client_gs.open("AI Sales Call Assistant").sheet1

def save_to_sheets(timestamp, text, sentiment, emotion, stop_reason):
    sheet.append_row([timestamp, text, sentiment, emotion, stop_reason])

print("🎤 Assistant started (stops if silence >5s)")

# 🔹 Step 1: Calibrate silence
print("\n🤫 Calibrating... stay quiet for 3s...")
calib = sd.rec(int(3 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
sd.wait()
baseline = np.linalg.norm(calib) / len(calib)

# ✅ Dynamic threshold adjustment
SILENCE_THRESHOLD = baseline * 1.2  
if SILENCE_THRESHOLD < 0.00005:   # fallback for weak mics
    SILENCE_THRESHOLD = 0.00005  

print(f"✅ Calibration done. Baseline={baseline:.6f}, Threshold={SILENCE_THRESHOLD:.6f}")

# 🔹 Step 2: Record continuously
recorded_audio = []
silence_counter = 0
start_time = time.time()
stop_reason = "User kept talking"

try:
    while True:  # continuous loop, no 3-min limit
        chunk = sd.rec(int(SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
        sd.wait()
        recorded_audio.append(chunk)

        elapsed = int(time.time() - start_time)
        volume = np.linalg.norm(chunk) / len(chunk)

        print(f"⏱ {elapsed}s | 🎚 Volume={volume:.6f}", end="\r")

        if volume < SILENCE_THRESHOLD:
            silence_counter += 1
            if silence_counter >= SILENCE_LIMIT:
                stop_reason = "Silent >5s"
                break
        else:
            silence_counter = 0

except KeyboardInterrupt:
    stop_reason = "Stopped by user"

print(f"\n🛑 Recording stopped: {stop_reason}")

# 🔹 Merge chunks into one array
recording = np.concatenate(recorded_audio, axis=0)

# 🔹 Save audio to wav
with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
    with wave.open(tmp_file.name, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((recording * 32767).astype(np.int16).tobytes())

    if stop_reason == "Silent >5s":
        text, sentiment_result, emotion_result = "Not Speaking", "N/A", "N/A"
    else:
        # Transcribe with Whisper on Groq
        with open(tmp_file.name, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f
            )
        text = transcription.text.strip()

        # Sentiment (force single word)
        sentiment = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role":"system","content":"Reply with only one word: Positive, Negative, or Neutral."},
                {"role":"user","content":text}
            ]
        )
        sentiment_result = sentiment.choices[0].message.content.strip().split()[0]

        # Emotion (force single word)
        emotion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role":"system","content":"Reply with only one word: Joy, Sadness, Anger, Fear, or Surprise."},
                {"role":"user","content":text}
            ]
        )
        emotion_result = emotion.choices[0].message.content.strip().split()[0]

# 🔹 Save results
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
file_exists = os.path.isfile(csv_file)
with open(csv_file, "a", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(["Timestamp", "Transcript", "Sentiment", "Emotion", "StopReason"])
    writer.writerow([timestamp, text, sentiment_result, emotion_result, stop_reason])

save_to_sheets(timestamp, text, sentiment_result, emotion_result, stop_reason)

print(f"\n📝 Transcript: {text}")
print(f"📊 Sentiment: {sentiment_result} | 🎭 Emotion: {emotion_result}")
print(f"✅ Results saved to CSV & Google Sheets")


























