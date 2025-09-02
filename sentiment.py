import numpy as np
import tempfile
import wave
from config import client, SAMPLE_RATE, CHANNELS

def save_wav(recording):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        with wave.open(tmp_file.name, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((recording * 32767).astype(np.int16).tobytes())
        return tmp_file.name

def analyze_audio(recording, stop_reason):
    if stop_reason == "Silent >5s":
        return "Not Speaking", "N/A", "N/A"

    wav_file = save_wav(recording)

    # ✅ Step 1: Transcription
    with open(wav_file, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f
        )
    text = transcription.text.strip()

    # ✅ Step 2: Sentiment
    sentiment = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # updated model
        messages=[
            {"role": "system", "content": "Reply with only one word: Positive, Negative, or Neutral."},
            {"role": "user", "content": text}   # now inside function
        ]
    )
    sentiment_result = sentiment.choices[0].message.content.strip().split()[0]

    # ✅ Step 3: Emotion
    emotion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Reply with only one word: Joy, Sadness, Anger, Fear, or Surprise."},
            {"role": "user", "content": text}   # also inside function
        ]
    )
    emotion_result = emotion.choices[0].message.content.strip().split()[0]

    return text, sentiment_result, emotion_result

