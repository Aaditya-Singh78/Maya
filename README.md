# Meta Voice Screening Agent

AI-powered first-round phone screening. Fully free — no paid APIs, no Twilio, no Vapi.

**Stack:** Groq Whisper (STT) · Groq Llama 3.1 8B (LLM) · edge-tts (TTS) · FastAPI

**Roles:** Software Engineer L5 · Technical Recruiter

**Languages:** English · Hindi · Hinglish (auto-detected, auto-switched)

**Cost:** $0 / month — all components are free tier

---

## Prerequisites

- Python 3.11+
- A **free** Groq API key → [console.groq.com](https://console.groq.com) → API Keys → Create Key
- A modern browser (Chrome / Edge / Safari) with microphone access

---

## Run Locally (2 minutes)

```bash
# 1. Clone / download the project
cd voice-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key
cp .env.example .env
# Open .env and paste your key: GROQ_API_KEY=gsk_...

# 4. Start the server
python main.py

# 5. Open in browser
# http://localhost:8000
```

---

## Share with Someone (ngrok — instant public URL)

```bash
# Install ngrok: https://ngrok.com/download (free account)
ngrok http 8000

# You'll get a URL like: https://abc123.ngrok-free.app
# Share that URL — anyone can open it and call the agent
```

---

## Deploy to Railway (permanent URL, free tier)

```bash
# 1. Push to GitHub
git init && git add . && git commit -m "init"
gh repo create meta-screening-agent --public --push

# 2. Go to railway.app → New Project → Deploy from GitHub
# 3. Select your repo
# 4. Add environment variable: GROQ_API_KEY = your_key
# 5. Deploy → get permanent URL like https://meta-screening-agent.up.railway.app
```

Railway gives $5 free credit/month — enough for ~500 screening calls.

---

## Deploy to Render (alternative, also free)

```bash
# 1. Push to GitHub (same as above)
# 2. render.com → New → Web Service → Connect repo
# 3. Runtime: Docker
# 4. Add env var: GROQ_API_KEY
# 5. Deploy
```

---

## How It Works

```
Browser mic → WebSocket → Groq Whisper STT (~150ms)
                       → Groq Llama 3.1 8B LLM (~200ms)
                       → edge-tts TTS (~60ms)
                       → Browser speaker
Total TTFS: ~430ms
```

### Call Flow
1. **Greeting** — verifies identity ("Am I speaking with [Name]?")
2. **Availability** — checks if now is a good time, offers callback if not
3. **Screening** — 4 role-specific questions, one at a time
4. **Next Steps** — informs candidate of 3–5 day timeline
5. **Close** — warm goodbye

### Language Switching
Whisper detects language per utterance. If Hindi or Hinglish is detected, the LLM switches response language and edge-tts switches to `hi-IN-SwaraNeural` voice. Switches are turn-level (not mid-utterance).

---

## How to Use the UI

1. Enter candidate name and select role
2. Click **Start Screening Call**
3. Wait for Maya's opening message to finish playing
4. **Hold the mic button** (or hold **Space**) to speak
5. **Release** to send your audio
6. Maya responds — wait for playback to finish before speaking again

On mobile: tap and hold the mic button.

---

## Free Tier Limits (Groq)

| Service | Free Limit |
|---------|-----------|
| Whisper large-v3 | 28,800 seconds audio/day (~480 minutes) |
| Llama 3.1 8B | 14,400 requests/day |

Each 7-minute call uses ~7 seconds of STT and ~15 LLM requests. The free tier supports **~60 full calls per day** with room to spare.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Free key from console.groq.com |
| `PORT` | No | Server port (default: 8000) |

---

## Project Structure

```
voice-agent/
├── main.py           # FastAPI server + WebSocket + full agent logic
├── static/
│   └── index.html    # Complete web UI (single file)
├── requirements.txt
├── Dockerfile
├── railway.json      # Railway deploy config
├── render.yaml       # Render deploy config
├── .env.example
└── README.md
```

---

## Troubleshooting

**"Couldn't hear that"** — audio too short or silent. Speak for at least 1-2 seconds.

**"Microphone access denied"** — check browser permissions. Chrome: click 🔒 in address bar → Microphone → Allow.

**"Connection error"** — server not running, or WebSocket blocked. Check `python main.py` is still running.

**Hindi voice sounds robotic** — edge-tts quality varies. For production, consider Suno Bark (fully open-source local TTS with better Hindi) or Microsoft Azure TTS (paid but excellent).

**Groq rate limit hit** — you've exceeded the free tier for the day. Resets at midnight UTC.

---

## Swap Components (all free)

| Component | Current | Free Alternative |
|-----------|---------|-----------------|
| STT | Groq Whisper | faster-whisper (local, no internet) |
| LLM | Groq Llama 3.1 8B | Ollama + Llama 3.1 (local) |
| TTS | edge-tts | Coqui TTS / Bark (local) |
| Server | FastAPI | Flask / any ASGI server |

To go fully local (no internet after setup): replace Groq with Ollama + faster-whisper + Coqui TTS. Needs ~8GB RAM.
