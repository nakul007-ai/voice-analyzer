# 🎙️ Voice Meeting Analyzer

An AI-powered meeting analysis application that converts meeting audio into text and automatically extracts useful information such as summaries, discussion points, decisions, action items, deadlines, testing plans, and pending questions.

The project uses **Whisper** for speech-to-text transcription and **Ollama with Llama 3.2** for local AI-based meeting analysis.

---

## 🚀 Features

- 🎙️ Record meeting audio directly from the browser
- 📁 Upload existing meeting audio
- 📝 Convert speech into text using Whisper
- 🤖 Analyze meetings using Llama 3.2
- 📌 Generate meeting summaries
- 💬 Extract important discussion points
- ✅ Identify action items
- 👤 Identify tasks assigned to people when available
- 📅 Detect deadlines and completion targets
- 🧪 Generate testing plans
- ❓ Identify pending questions
- 📄 Generate meeting analysis reports as PDF
- 💾 Save transcripts and analysis data
- 🏠 Run AI processing locally using Ollama

---

# 🧠 How It Works

The application follows this workflow:

```text
                Meeting Audio
                     │
                     ▼
              Browser Recording
                     │
                     ▼
                  Flask
                     │
                     ▼
                 Whisper
                     │
                     ▼
                Transcript
                     │
                     ▼
              Ollama / Llama 3.2
                     │
                     ▼
              Meeting Analysis
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
     Summary     Action Items   Decisions
        │            │            │
        └────────────┼────────────┘
                     ▼
              PDF Report
