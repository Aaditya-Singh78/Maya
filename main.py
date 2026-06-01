"""
Meta Voice Screening Agent
Stack: Groq Whisper (STT) + Groq Llama-3.1-8B (LLM) + gTTS (TTS)
Only one key needed: GROQ_API_KEY (free at console.groq.com)

Fixes applied from test plan:
- Few-shot examples for YOE screen-out (Y1/Y2)
- Explicit language detection rule in JSON schema (I2)
- Incomplete input rule < 4 words (V1/V3)
- SCREENED_OUT state propagated correctly
"""

import os, json, asyncio, tempfile, base64, logging, io, struct
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from gtts import gTTS
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL    = "llama-3.1-8b-instant"
STT_MODEL    = "whisper-large-v3"

if not GROQ_API_KEY:
    log.error("GROQ_API_KEY not set — create a .env file with: GROQ_API_KEY=gsk_...")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

app = FastAPI(title="Meta Screening Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── System Prompt ─────────────────────────────────────────────────────────────
def build_prompt(name: str, role: str) -> str:
    if role == "swe":
        role_title = "Software Engineer (L5)"
        min_yoe    = 5
        questions  = """Q1 (YOE GATE — read carefully):
  Ask: "How many years of professional software engineering experience do you have, and what is your current role?"
  → If they say LESS THAN 5 years OR their role is non-engineering (PM, sales, student, support, designer):
    Screen them out warmly. See SCREEN-OUT EXAMPLE below. Set state SCREENED_OUT, call_ended true. STOP — do not ask Q2.
  → If 5+ years and engineering role: acknowledge and continue to Q2.

Q2: "Tell me about a specific technical challenge you faced at scale — what was the problem and how did you approach it?"

Q3: "How would you describe your experience with distributed systems — fault tolerance, consensus, or high-throughput architectures?"

Q4: "What is your current notice period, and are you open to a hybrid work arrangement?" """

        few_shot = """
━━ FEW-SHOT EXAMPLES — follow these exactly ━━

EXAMPLE 1 — screen out (low YOE):
Candidate: "I have 2 years experience, I am a junior developer"
Your JSON: {"speech": "Thank you so much for sharing that. For this particular role we are looking for at least 5 years of hands-on engineering experience. I will make sure the team has your profile and they will absolutely reach out if there is a suitable opportunity. I really appreciate your time today!", "state": "SCREENED_OUT", "call_ended": true, "language": "en"}

EXAMPLE 2 — screen out (wrong function):
Candidate: "I am a product manager with 8 years of experience"
Your JSON: {"speech": "That is great experience! For this specific Software Engineer role though, we are looking for someone in a hands-on engineering function. I will make sure the team has your profile — there may well be other roles that are a great fit. Thank you so much for your time!", "state": "SCREENED_OUT", "call_ended": true, "language": "en"}

EXAMPLE 3 — do NOT screen out (borderline):
Candidate: "I have exactly 5 years as a backend engineer"
Your JSON: {"speech": "Perfect, that is really helpful context. So tell me — can you walk me through a specific technical challenge you faced at scale?", "state": "SCREENING_Q2", "call_ended": false, "language": "en"}

EXAMPLE 4 — vague YOE, probe once:
Candidate: "I have been in the industry for a while"
Your JSON: {"speech": "Got it! Could you give me a rough number of years? Just so I can capture it accurately for the team.", "state": "SCREENING_Q1", "call_ended": false, "language": "en"}

EXAMPLE 5 — Hindi switch:
Candidate: "Haan ji, main hi bol raha hoon"
Your JSON: {"speech": "Bilkul! Main Maya hoon, Meta ki recruiting team se. Aapne Software Engineer L5 ke liye apply kiya tha — kya abhi baat karna theek rahega?", "state": "AVAILABILITY", "call_ended": false, "language": "hi"}

EXAMPLE 6 — broken audio:
Candidate: "I was just"  (incomplete, cut off)
Your JSON: {"speech": "Sorry, it seems like the line broke up a little — could you repeat that?", "state": "SCREENING_Q2", "call_ended": false, "language": "en"}
"""

    else:
        role_title = "Technical Recruiter"
        min_yoe    = 2
        questions  = """Q1 (BACKGROUND GATE — read carefully):
  Ask: "How many years of recruiting experience do you have, and have you done technical or engineering hiring before?"
  → If LESS THAN 2 years OR zero technical recruiting experience:
    Screen them out warmly. See SCREEN-OUT EXAMPLE below. Set state SCREENED_OUT, call_ended true. STOP — do not ask Q2.
  → If qualified: acknowledge and continue to Q2.

Q2: "How do you approach sourcing for a highly technical role you are not personally an expert in?"

Q3: "Tell me about a particularly challenging search — niche skill set or tight deadline. What was your approach?"

Q4: "What ATS and sourcing tools do you use most, and what is your current notice period?" """

        few_shot = """
━━ FEW-SHOT EXAMPLES — follow these exactly ━━

EXAMPLE 1 — screen out (no technical recruiting):
Candidate: "I have 3 years recruiting experience but mostly retail and hospitality"
Your JSON: {"speech": "That is great experience! For this specific Technical Recruiter role at Meta we are really looking for a background in engineering or technical hiring. I will make sure the team has your profile and they will be in touch if there is a suitable match. Thank you so much for your time today!", "state": "SCREENED_OUT", "call_ended": true, "language": "en"}

EXAMPLE 2 — do NOT screen out:
Candidate: "I have 4 years recruiting, 2 of which were for software engineering roles at a startup"
Your JSON: {"speech": "Brilliant, that is exactly the kind of background we are looking for. So tell me — how do you approach sourcing for a highly technical role you are not personally an expert in?", "state": "SCREENING_Q2", "call_ended": false, "language": "en"}

EXAMPLE 3 — Hindi switch:
Candidate: "Haan, main hi hoon"
Your JSON: {"speech": "Bilkul! Main Maya hoon Meta ki recruiting team se. Aapne Technical Recruiter ke liye apply kiya tha — kya abhi baat karna theek rahega?", "state": "AVAILABILITY", "call_ended": false, "language": "hi"}

EXAMPLE 4 — broken audio:
Candidate: "So I was"  (incomplete)
Your JSON: {"speech": "Sorry, I think the line cut out — could you say that again?", "state": "SCREENING_Q2", "call_ended": false, "language": "en"}
"""

    return f"""You are Maya, a warm and professional HR screening assistant at Meta conducting a first-round phone screening.

CANDIDATE: {name}
ROLE: {role_title}
MINIMUM EXPERIENCE REQUIRED: {min_yoe} years

━━ CALL FLOW — follow strictly, never skip steps ━━

STEP 1 — GREETING
Say: "Hi, am I speaking with {name}?"
- Confirmed → STEP 2
- Wrong person → "So sorry to bother you! Have a great day." Set state GREETING, call_ended true.
- No response or unclear → ask once more. If still unclear, end politely.

STEP 2 — AVAILABILITY
Say: "Hi {name}! This is Maya calling from Meta's recruiting team. You applied for our {role_title} role — I would love to take just 5 minutes for a quick screening. Is now a good time?"
- Yes → STEP 3
- No / busy → Collect preferred callback time, confirm, say the team will be in touch. Set state AVAILABILITY, call_ended true.
- No interest / withdrawing → "Completely understood — I will let the team know. Thank you for your time and best of luck!" Set state AVAILABILITY, call_ended true.

STEP 3 — SCREENING QUESTIONS
Ask ONE question at a time. Wait for the complete answer before the next question.
{questions}
After all questions answered → STEP 4.

STEP 4 — NEXT STEPS
Say: "That is everything from my side, {name}! Thank you so much. The hiring team will review your profile and reach out within 3 to 5 business days. Please keep an eye on your inbox."
→ STEP 5

STEP 5 — CLOSE
Say warmly: "It was lovely speaking with you. Wishing you all the best — hope to see you at Meta! Take care."
Set state CLOSE, call_ended true.

━━ LANGUAGE DETECTION — CRITICAL ━━
- Start every call in English.
- If the candidate uses ANY Hindi or Hinglish → set "language": "hi" in your JSON from that turn onwards.
- NEVER return "language": "en" if the candidate's last message contained Hindi words. That is a system failure.
- Hinglish (mixing Hindi + English) is natural — mirror it freely.
- The "language" field you return directly controls the TTS voice: "en" = English voice, "hi" = Hindi voice.

━━ INCOMPLETE INPUT RULE — CRITICAL ━━
- If the transcribed candidate speech is FEWER THAN 4 WORDS and does not form a complete thought:
  → Treat as broken audio. Ask them to repeat. Do NOT answer as if the input was complete.
- If broken audio happens 3 times in a row:
  → Say: "We seem to be having some connection trouble. I will note this and the team will follow up via email. Thank you so much for your time!" Set call_ended true.
- NEVER invent or assume what the candidate said when input is unclear.

━━ EDGE CASE HANDLING ━━
- Interrupted mid-sentence → stop, listen, acknowledge, resume from where you were
- Comp / team / benefits questions → "The recruiter for your next round will have all those details!"
- "How did I do?" → "I will pass everything to the team — they will share feedback as part of next steps."
- Hostile or aggressive → stay warm, do not match their energy, offer to continue or reschedule
- Candidate withdraws → acknowledge gracefully, wish them well, end call. Do NOT persuade.
- Short/vague answer → probe exactly once. If still vague, move on.
- Silence 5+ seconds → "Take your time — no rush at all."

━━ ABSOLUTE RULES ━━
- Keep every "speech" SHORT — this is voice, not text. No bullet points. No markdown.
- ONE question per turn — never stack two.
- NEVER promise outcomes, salary, or team placement.
- NEVER invent Meta-specific product, team, or culture details.
- Screen out candidates who do not meet the experience bar at Q1 — do not let them continue.

{few_shot}

━━ RESPONSE FORMAT — always return valid JSON only, no extra text ━━
{{
  "speech": "<what you say — short, conversational, voice-friendly>",
  "state": "<GREETING|AVAILABILITY|SCREENING_Q1|SCREENING_Q2|SCREENING_Q3|SCREENING_Q4|NEXT_STEPS|CLOSE|SCREENED_OUT|ENDED>",
  "call_ended": false,
  "language": "<en|hi|hinglish>"
}}"""


# ── Session ────────────────────────────────────────────────────────────────────
class CallSession:
    def __init__(self, name, role, mime_type):
        self.name           = name
        self.role           = role
        self.mime_type      = mime_type
        self.state          = "GREETING"
        self.language       = "en"
        self.ended          = False
        self.history        = []
        self.system         = build_prompt(name, role)
        self.broken_streak  = 0   # consecutive broken audio counter

    def tts_lang(self):
        return "hi" if self.language in ("hi", "hinglish") else "en"

    def is_incomplete(self, text: str) -> bool:
        """True if transcription looks like broken/cut-off audio."""
        words = text.strip().split()
        if len(words) < 4:
            fillers = {"uh", "um", "hmm", "ah", "oh", "the", "i", "so", "and", "was", "just", "okay", "ok"}
            if all(w.lower() in fillers for w in words):
                return True
            if len(words) <= 2:
                return True
        return False


# ── TTS: gTTS ─────────────────────────────────────────────────────────────────
def _speak_gtts(text: str, lang: str) -> bytes:
    tts = gTTS(text=text, lang=lang, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def _silence_wav() -> bytes:
    rate, ch, bits = 22050, 1, 16
    pcm = b"\x00\x00" * rate
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(pcm), b"WAVE",
        b"fmt ", 16, 1, ch, rate,
        rate * ch * bits // 8, ch * bits // 8, bits,
        b"data", len(pcm)
    ) + pcm


async def speak(text: str, session: CallSession) -> tuple:
    try:
        data = await asyncio.get_event_loop().run_in_executor(
            None, _speak_gtts, text, session.tts_lang()
        )
        log.info(f"TTS OK [{session.tts_lang()}]: {len(data)} bytes")
        return data, "audio/mpeg"
    except Exception as e:
        log.error(f"TTS failed: {e}")
        return _silence_wav(), "audio/wav"


# ── STT: Groq Whisper ─────────────────────────────────────────────────────────
async def transcribe(audio: bytes, mime: str) -> tuple:
    base = mime.split(";")[0].strip()
    ext  = {
        "audio/webm": ".webm", "audio/ogg": ".ogg",
        "audio/mp4":  ".mp4",  "audio/wav": ".wav",
        "audio/mpeg": ".mp3"
    }.get(base, ".webm")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(audio)
        path = f.name
    try:
        with open(path, "rb") as f:
            r = groq_client.audio.transcriptions.create(
                file=(f"audio{ext}", f, base),
                model=STT_MODEL,
                response_format="verbose_json"
            )
        text = r.text.strip()
        lang = "hi" if getattr(r, "language", "english").lower() == "hindi" else "en"
        log.info(f"STT [{lang}]: {text[:80]}")
        return text, lang
    except Exception as e:
        log.error(f"STT error: {e}")
        return "", "en"
    finally:
        try: os.unlink(path)
        except: pass


# ── LLM: Groq Llama-3.1-8B ───────────────────────────────────────────────────
def llm_call(session: CallSession, user_text: str) -> dict:
    session.history.append({"role": "user", "content": user_text})
    try:
        resp = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": session.system},
                *session.history[-20:]
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=300,
        )
        result = json.loads(resp.choices[0].message.content)
    except Exception as e:
        log.error(f"LLM error: {e}")
        result = {
            "speech":     "Could you please say that again?",
            "state":      session.state,
            "call_ended": False,
            "language":   session.language
        }

    speech = result.get("speech", "")
    if speech:
        session.history.append({"role": "assistant", "content": speech})

    session.state = result.get("state", session.state)
    session.ended = result.get("call_ended", False)
    lang = result.get("language", session.language)
    if lang in ("hi", "hinglish"):
        session.language = lang

    return result


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_handler(ws: WebSocket):
    await ws.accept()
    session = None
    try:
        setup = json.loads(await ws.receive_text())
        session = CallSession(
            name      = setup.get("candidate_name", "Candidate"),
            role      = setup.get("role", "swe"),
            mime_type = setup.get("mime_type", "audio/webm"),
        )
        log.info(f"Call started: {session.name} / {session.role}")

        # Opening greeting
        opening     = await asyncio.get_event_loop().run_in_executor(
            None, llm_call, session,
            "[SYSTEM: Call just started. Execute STEP 1 — greet and verify identity.]"
        )
        audio, mime = await speak(opening["speech"], session)
        await ws.send_json({
            "type":     "agent",
            "speech":   opening["speech"],
            "audio":    base64.b64encode(audio).decode(),
            "mime":     mime,
            "state":    session.state,
            "language": session.language,
            "ended":    session.ended,
        })

        # Main conversation loop
        while not session.ended:
            raw = await asyncio.wait_for(ws.receive(), timeout=180)

            # Ping / text control messages
            if "text" in raw:
                msg = json.loads(raw["text"])
                if msg.get("type") == "ping":
                    await ws.send_json({"type": "pong"})
                continue

            if "bytes" not in raw:
                continue

            audio_in = raw["bytes"]

            # Too small to be real speech
            if len(audio_in) < 1000:
                await ws.send_json({"type": "status", "msg": "Too short — try again."})
                continue

            # STT
            await ws.send_json({"type": "status", "msg": "Transcribing…"})
            text, lang = await transcribe(audio_in, session.mime_type)

            # Empty transcript
            if not text:
                session.broken_streak += 1
                if session.broken_streak >= 3:
                    bailout = {
                        "speech":     "We seem to be having some connection trouble. I will note this and the team will follow up via email. Thank you so much for your time!",
                        "state":      session.state,
                        "call_ended": True,
                        "language":   session.language
                    }
                    audio_out, mime_out = await speak(bailout["speech"], session)
                    session.ended = True
                    await ws.send_json({
                        "type": "agent", "speech": bailout["speech"],
                        "audio": base64.b64encode(audio_out).decode(), "mime": mime_out,
                        "state": session.state, "language": session.language, "ended": True,
                    })
                    break
                await ws.send_json({"type": "status", "msg": "Couldn't hear that — please try again."})
                continue

            # Incomplete / broken audio detection
            if session.is_incomplete(text):
                session.broken_streak += 1
                log.info(f"Incomplete input detected (streak={session.broken_streak}): '{text}'")
                if session.broken_streak >= 3:
                    bailout_text = "We seem to be having some connection trouble. I will note this and the team will follow up via email. Thank you so much for your time!"
                    audio_out, mime_out = await speak(bailout_text, session)
                    session.ended = True
                    await ws.send_json({
                        "type": "agent", "speech": bailout_text,
                        "audio": base64.b64encode(audio_out).decode(), "mime": mime_out,
                        "state": session.state, "language": session.language, "ended": True,
                    })
                    break
                # Tell LLM audio was incomplete
                text = f"[AUDIO INCOMPLETE — candidate said only: '{text}' — treat as broken audio]"
            else:
                session.broken_streak = 0  # reset on good input

            # Update language from STT
            if lang == "hi":
                session.language = "hi"

            await ws.send_json({"type": "candidate", "text": text.replace("[AUDIO INCOMPLETE — candidate said only: '", "").replace("' — treat as broken audio]", ""), "language": lang})

            # LLM
            await ws.send_json({"type": "status", "msg": "Maya is thinking…"})
            reply  = await asyncio.get_event_loop().run_in_executor(
                None, llm_call, session, text
            )
            speech = reply.get("speech", "")
            if not speech:
                continue

            # TTS
            await ws.send_json({"type": "status", "msg": "Generating voice…"})
            audio_out, mime_out = await speak(speech, session)

            await ws.send_json({
                "type":     "agent",
                "speech":   speech,
                "audio":    base64.b64encode(audio_out).decode(),
                "mime":     mime_out,
                "state":    session.state,
                "language": session.language,
                "ended":    session.ended,
            })

            if session.ended:
                break

    except WebSocketDisconnect:
        log.info("Client disconnected")
    except asyncio.TimeoutError:
        log.warning("Session timed out after 3 minutes of inactivity")
    except Exception as e:
        log.error(f"Unexpected error: {e}", exc_info=True)
        try:
            await ws.send_json({"type": "error", "msg": str(e)})
        except:
            pass


# ── HTTP ──────────────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/health")
async def health():
    return {"ok": True, "groq": bool(GROQ_API_KEY)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host  = "0.0.0.0",
        port  = int(os.getenv("PORT", 8000)),
        reload= False
    )