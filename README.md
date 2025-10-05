# 🎤🤖 AI Sales Call Assistant  

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

AI Sales Call Assistant is a **real-time AI-powered tool** that helps sales agents transcribe, analyze, and summarize calls.  
It records live audio 🎙️, transcribes it using **Groq Whisper**, analyzes sentiment & emotion using **Groq LLaMA3**, and stores results in **Google Sheets**.  
A built-in **Streamlit dashboard** allows agents to log in, review summaries, and generate **AI Post-Call Reports**.

---
## 📁 Project Structure
```graphql
ai-sales-call-assistant/
├── app_streamlit.py          # Streamlit dashboard
├── groq_assistant.py         # CLI voice assistant
├── config.py                 # Audio & session settings
├── speech_to_text.py         # Handles microphone input
├── google_sheets.py          # Integration with Google Sheets
├── .streamlit/
│   └── secrets.toml          # Contains Groq API key
├── credentials.json          # Google Service Account file
├── requirements.txt
└── README.md
```
## ✨ Features

### 🎧 Core Assistant (CLI)
- 🕵️ **Continuous Speech Capture** — Automatically stops after 5 seconds of silence.  
- ✍️ **Speech-to-Text** — High-accuracy English transcription via `whisper-large-v3`.  
- 💬 **Sentiment Analysis** — Classifies as Positive / Negative / Neutral.  
- 🎭 **Emotion Detection** — Detects Joy, Sadness, Anger, Fear, or Surprise.  
- 📊 **Google Sheets Logging** — Auto-stores transcript, timestamp, sentiment & emotion.  
- ⚙️ **Reusable Modules** — Designed for integration into other AI systems.

### 💻 Streamlit Dashboard (Agent Portal)
- 🔐 **Agent Login** — Secure login (username/password).  
- 📱 **Search Summaries** — Search by customer phone number.  
- 🧠 **AI Summary Generator** — Produces a professional post-call report with:
  - 💬 **Overall Sentiment**
  - 🎯 **Customer Intent**
  - 🧩 **Key Topics**
  - ⚠️ **Objections**
  - ✅ **Resolutions**
  - 📝 **Next Steps**
  - 🔁 **Recommended Follow-up**
- 🚪 **Auto Logout** — Automatically logs out when user switches tabs.  
- 🧾 **Clean Card Layout** — Beautiful UI with neatly formatted summaries.

---

## 🧠 Technologies Used

| Area                     | Technology / Library Used                             |
|--------------------------|-------------------------------------------------------|
| **Programming Language** | Python 3.11                                           |
| **Framework**            | Streamlit                                            |
| **Speech-to-Text**       | Groq Whisper (`whisper-large-v3`)                    |
| **AI Summarization**     | Groq LLaMA 3.3 (70B Versatile)                       |
| **AI API**               | Groq Cloud API                                       |
| **Audio Recording**      | sounddevice, numpy                                   |
| **Data Storage**         | Google Sheets via gspread & oauth2client             |
| **Data Handling**        | pandas                                               |
| **Secrets Management**   | Streamlit Secrets.toml                               |
| **Frontend Styling**     | Streamlit HTML/CSS Styling                           |
| **Deployment Ready**     | Works on local machine and Streamlit Cloud           |

---

## 🧰 Tech Stack Overview

| Feature              | Technology Used                            |
|----------------------|--------------------------------------------|
| Transcription        | **Groq Whisper (whisper-large-v3)**        |
| AI Summaries         | **Groq LLaMA3.3 70B Versatile**            |
| Dashboard UI         | **Streamlit**                              |
| Data Storage         | **Google Sheets (via gspread)**            |
| Audio Processing     | **sounddevice**, **numpy**                 |
| Secrets Management   | **Streamlit Secrets.toml**                 |

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/ai-sales-call-assistant.git
cd ai-sales-call-assistant
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
# OR manually:
pip install streamlit sounddevice numpy gspread oauth2client groq pandas
```
### 3️⃣ Add API Credentials
🔑 Groq API Key
Get your API key from [Groq Console](https://console.groq.com/keys)
```bash
export GROQ_API_KEY="your_api_key_here"   # Linux/Mac
setx GROQ_API_KEY "your_api_key_here"     # Windows
```

### 📄 Google Sheets Service Account

1. Create a Service Account in Google Cloud Console
2. Download the JSON credentials
3. Save it as credentials.json in the project root
4. Share your target Google Sheet with the service account email

### 🔐 Streamlit Secrets
Create a file: **.streamlit/secrets.toml**
```toml
GROQ_API_KEY = "your_api_key_here"
```

### ▶️ Usage:
1. 🧠 CLI Assistant (Voice Recorder)
   ```bash
   python groq_assistant.py
   ```
2. 🎙️ Speak naturally into your mic
3. 🕒 Stops automatically after 5 seconds of silence
4. 📊 Data is logged to Google Sheets automatically ✅

### 💻 Streamlit Dashboard
```bash
streamlit run app_streamlit.py
```
Then open in browser: ([Link](http://localhost:8501))

### 🔐 Login
```yaml
Username: agent  
Password: 1234
```
🧭 Steps

1. Enter a customer phone number
2. View all saved summaries
3. Click 🤖 Generate AI Summary to get a structured post-call report

### 📋 Example AI Summary Output
```vbnet
💬 **Overall Sentiment** The overall sentiment was positive with occasional confusion.  
🎯 **Customer Intent** Interested in exploring enterprise pricing and features.  
🧩 **Key Topics** Discussed onboarding, SLA, support, and integration timelines.  
⚠️ **Objections** Concern about compatibility with current CRM.  
✅ **Resolutions** Scheduled follow-up with technical team.  
📝 **Next Steps** Send pricing sheet and set up demo call.  
🔁 **Recommended Follow-up** Weekly update until closure.
```

### 📦 Requirements
Here’s a quick reference for your requirements.txt:
```nginx
streamlit
sounddevice
numpy
pandas
gspread
oauth2client
groq
```
💡 Powered by Groq Whisper + LLaMA3 and Streamlit
