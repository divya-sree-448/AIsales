# AI Sales Call Assistant 🎤🤖

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

AI Sales Call Assistant is a **real-time assistant** for transcribing and analyzing sales calls. It records live audio from your microphone, detects silence, transcribes speech using **Groq Whisper**, and performs **sentiment & emotion analysis** using Groq LLaMA3 models. Results are automatically stored in **Google Sheets** for review and analytics.

## Features
- 🎙 **Continuous Speech Capture**: Stops automatically after 5 seconds of silence.
- 📝 **Speech-to-Text**: Uses Groq Whisper (`whisper-large-v3`) for high-accuracy English transcription.
- 📊 **Sentiment Analysis**: Single-word classification (Positive, Negative, Neutral).
- 🎭 **Emotion Detection**: Single-word classification (Joy, Sadness, Anger, Fear, Surprise).
- 🗂 **Google Sheets Integration**: Timestamped transcript, sentiment, emotion, and stop reason.
- ⚡ **Modular & Reusable**: Python module design for easy integration or extension.

## Technologies Used
- Python 3
- [Groq Whisper](https://groq.com/) for transcription
- [Groq LLaMA3](https://groq.com/) for sentiment & emotion analysis
- SoundDevice & NumPy for audio recording
- Google Sheets API via `gspread`
- Temporary `.wav` handling using `tempfile` and `wave`

## Setup & Installation
1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-sales-call-assistant.git
cd ai-sales-call-assistant

```
##
2. **Install dependencies**:
```bash
pip install -r requirements.txt
# OR individually
pip install sounddevice numpy gspread oauth2client groq

```
3. **ADD API Credentials**:
- Groq API Key: Set as environment variable
```bash
export GROQ_API_KEY="your_api_key_here"  # Linux/Mac
setx GROQ_API_KEY "your_api_key_here"     # Windows
```
- Google Sheets credentials: Place your service account JSON as (credentials.json) File

## Usage
- Run the assistant :
```bash
python groq_assistant.py
```
- Speak naturally into your microphone.
- The assistant will stop if you are silent for 5 seconds or if you press Ctrl+C.
- Transcript, sentiment, emotion, and stop reason will be saved automatically to Google Sheets.
   

