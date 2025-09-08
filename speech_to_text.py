import sounddevice as sd
import numpy as np
import time
from typing import List, Tuple, Optional
from config import SAMPLE_RATE, CHANNELS, SILENCE_LIMIT

def calibrate_silence() -> float:
    """
    Record 3s of ambient audio and return a threshold slightly above baseline.
    """
    print("\n Calibrating... stay quiet for 3s...")
    calib = sd.rec(int(3 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
    sd.wait()
    baseline = np.linalg.norm(calib) / max(len(calib), 1)

    threshold = max(baseline * 1.2, 0.00005)
    print(f" Calibration done. Baseline={baseline:.6f}, Threshold={threshold:.6f}")
    return threshold


def record_until_silence(
    SILENCE_THRESHOLD: float,
    stop_event: Optional[object] = None,
    max_duration_s: int = 3600,
) -> Tuple[List[np.ndarray], str]:
    """
    Record 1-second chunks until either:
      - continuous silence for SILENCE_LIMIT seconds, OR
      - stop_event is set from the UI, OR
      - max_duration_s is exceeded.

    Returns (list_of_chunks, stop_reason).
    """
    recorded_audio: List[np.ndarray] = []
    silence_counter = 0
    start_time = time.time()
    stop_reason = "User kept talking"

    try:
        while True:
            # time cap
            if (time.time() - start_time) > max_duration_s:
                stop_reason = "Time Limit Exceeded"
                break

            # always capture a chunk first so we never return empty on quick Stop
            chunk = sd.rec(int(SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
            sd.wait()
            recorded_audio.append(chunk)

            elapsed = int(time.time() - start_time)
            volume = np.linalg.norm(chunk) / max(len(chunk), 1)
            print(f" {elapsed:02d}s | Volume={volume:.6f}", end="\r")

            # if user pressed Stop, exit *after* we captured this chunk
            if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
                stop_reason = "User Stopped"
                break

            # silence detection
            if volume < SILENCE_THRESHOLD:
                silence_counter += 1
                if silence_counter >= SILENCE_LIMIT:
                    stop_reason = f"Silent >{SILENCE_LIMIT}s"
                    break
            else:
                silence_counter = 0

    except KeyboardInterrupt:
        stop_reason = "Stopped by user"

    print(f"\n Recording stopped: {stop_reason}")

    # safety: if nothing captured (very fast stop), grab a tiny 0.5s chunk
    if len(recorded_audio) == 0:
        tiny = sd.rec(int(0.5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
        sd.wait()
        if tiny is not None and len(tiny) > 0:
            recorded_audio.append(tiny)

    return recorded_audio, stop_reason






