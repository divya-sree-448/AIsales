import sounddevice as sd
import numpy as np
import time
from config import SAMPLE_RATE, CHANNELS, SILENCE_LIMIT

def calibrate_silence():
    print("\n Calibrating... stay quiet for 3s...")
    calib = sd.rec(int(3 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
    sd.wait()
    baseline = np.linalg.norm(calib) / len(calib)

    threshold = baseline * 1.2
    if threshold < 0.00005:
        threshold = 0.00005
    print(f" Calibration done. Baseline={baseline:.6f}, Threshold={threshold:.6f}")
    return threshold

def record_until_silence(SILENCE_THRESHOLD):
    recorded_audio = []
    silence_counter = 0
    start_time = time.time()
    stop_reason = "User kept talking"

    try:
        while True:
            chunk = sd.rec(int(SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
            sd.wait()
            recorded_audio.append(chunk)

            elapsed = int(time.time() - start_time)
            volume = np.linalg.norm(chunk) / len(chunk)

            print(f" {elapsed}s | Volume={volume:.6f}", end="\r")

            if volume < SILENCE_THRESHOLD:
                silence_counter += 1
                if silence_counter >= SILENCE_LIMIT:
                    stop_reason = "Silent >5s"
                    break
            else:
                silence_counter = 0

    except KeyboardInterrupt:
        stop_reason = "Stopped by user"

    print(f"\n Recording stopped: {stop_reason}")
    return recorded_audio, stop_reason
