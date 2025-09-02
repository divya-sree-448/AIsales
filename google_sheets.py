# import csv
# import os
# import datetime
# from config import sheet, CSV_FILE

# def save_to_sheets(timestamp, text, sentiment, emotion, stop_reason):
#     sheet.append_row([timestamp, text, sentiment, emotion, stop_reason])

# def save_to_csv(text, sentiment_result, emotion_result, stop_reason):
#     timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     file_exists = os.path.isfile(CSV_FILE)
#     with open(CSV_FILE, "a", newline="", encoding="utf-8") as csvfile:
#         writer = csv.writer(csvfile)
#         if not file_exists:
#             writer.writerow(["Timestamp", "Transcript", "Sentiment", "Emotion", "StopReason"])
#         writer.writerow([timestamp, text, sentiment_result, emotion_result, stop_reason])
#     return timestamp

import csv
import os
import datetime
from config import sheet, CSV_FILE

HEADERS = ["Timestamp", "Transcript", "Sentiment", "Emotion", "StopReason"]

def ensure_headers():
    """Make sure Google Sheet has headers in the first row."""
    values = sheet.get_all_values()
    if not values:  # completely empty sheet
        sheet.insert_row(HEADERS, 1)   # ✅ insert headers at row 1
    else:
        # if first row is not our headers, replace it
        if values[0] != HEADERS:
            sheet.update('A1:E1', [HEADERS])

def save_to_sheets(timestamp, text, sentiment, emotion, stop_reason):
    ensure_headers()
    sheet.append_row([timestamp, text, sentiment, emotion, stop_reason])

def save_to_csv(text, sentiment_result, emotion_result, stop_reason):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(HEADERS)   # ✅ write header to CSV if missing
        writer.writerow([timestamp, text, sentiment_result, emotion_result, stop_reason])
    return timestamp

