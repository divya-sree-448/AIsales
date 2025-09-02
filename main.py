import numpy as np
from speech_to_text import calibrate_silence, record_until_silence
from sentiment import analyze_audio
from google_sheets import save_to_sheets, save_to_csv

print(" Assistant started (stops if silence >5s)")

# Step 1: Calibrate
SILENCE_THRESHOLD = calibrate_silence()

# Step 2: Record
recorded_audio, stop_reason = record_until_silence(SILENCE_THRESHOLD)

# Step 3: Merge chunks
recording = np.concatenate(recorded_audio, axis=0)

# Step 4: Analyze
text, sentiment_result, emotion_result = analyze_audio(recording, stop_reason)

# Step 5: Save
timestamp = save_to_csv(text, sentiment_result, emotion_result, stop_reason)
save_to_sheets(timestamp, text, sentiment_result, emotion_result, stop_reason)

# Step 6: Print results
print(f"\n Transcript: {text}")
print(f" Sentiment: {sentiment_result} | Emotion: {emotion_result}")
print(f" Results saved to CSV & Google Sheets")
