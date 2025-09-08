import numpy as np
import time
from speech_to_text import calibrate_silence, record_until_silence
from sentiment import analyze_audio
from google_sheets import save_to_sheets

def main():
    print("🎤 Assistant started (stops if silence >5s)")

    # Step 1: Calibrate
    SILENCE_THRESHOLD = calibrate_silence()

    # Step 2: Record
    recorded_audio, stop_reason = record_until_silence(SILENCE_THRESHOLD)

    # Step 3: Merge chunks (combine all audio chunks into one array)
    recording = np.concatenate(recorded_audio, axis=0)

    # Step 4: Analyze the recorded audio
    text, sentiment_result, emotion_result = analyze_audio(recording, stop_reason)

    # Step 5: Save results to Google Sheets
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    save_to_sheets(timestamp, text, sentiment_result, emotion_result, stop_reason)

    # Step 6: Print results
    print(f"\n📝 Transcript: {text}")
    print(f"📊 Sentiment: {sentiment_result} | 🎭 Emotion: {emotion_result}")
    print(f"✅ Results saved to Google Sheets")

if __name__ == "__main__":
    main()






