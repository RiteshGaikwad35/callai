"""
Multilingual voice AI calling platform — single-file deployment.

Everything lives here: config, encrypted Twilio credential storage,
call/session state, STT (Groq hosted Whisper — no local model, so it
runs on free-tier RAM), LLM (Groq), TTS (edge-tts, free, no API key),
the Twilio call trigger, all REST/webhook routes, and the frontend
control panel (served as embedded HTML/CSS/JS at "/").

ENV VARS REQUIRED
    GROQ_API_KEY                 free at console.groq.com
    PUBLIC_HOST                  your public domain, no scheme, e.g. myapp.onrender.com
    CREDENTIALS_ENCRYPTION_KEY   generate with:
        python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

OPTIONAL ENV VARS
    LLM_MODEL           default: llama-3.3-70b-versatile
    LOG_LEVEL            default: INFO
    CREDENTIALS_PATH     default: data/twilio_credentials.enc

RUN LOCALLY
    pip install -r requirements.txt
    export GROQ_API_KEY=...  PUBLIC_HOST=...  CREDENTIALS_ENCRYPTION_KEY=...
    uvicorn app:app --host 0.0.0.0 --port 8000

Twilio credentials (Account SID / Auth Token / phone number) are entered
through the web UI, not env vars — they're encrypted at rest separately
from everything else. Open http://localhost:8000 (or your deployed URL)
to use the control panel.
"""

import asyncio
import audioop
import base64
import contextlib
import io
import json
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import webrtcvad
from cryptography.fernet import Fernet, InvalidToken
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from gtts import gTTS
from groq import APIError, Groq
from pydantic import BaseModel, Field
from pydub import AudioSegment
from twilio.rest import Client as TwilioClient

# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    public_host: str
    llm_model: str = "llama-3.3-70b-versatile"
    llm_max_tokens: int = 120
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 0.5
    vad_aggressiveness: int = 2
    silence_ms_to_end_turn: int = 700
    session_ttl_seconds: int = 600
    call_context_ttl_seconds: int = 3600
    max_history_turns: int = 12
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        public_host = os.environ.get("PUBLIC_HOST")
        if not public_host:
            raise RuntimeError("PUBLIC_HOST is not set (e.g. myapp.onrender.com, no https://).")
        return cls(
            groq_api_key=groq_key,
            public_host=public_host,
            llm_model=os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile"),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )


TTS_LANG_MAP = {
    "en": "en",
    "hi": "hi",
    "ta": "ta",
}
DEFAULT_TTS_LANG = "en"

BASE_SYSTEM_INSTRUCTIONS = (
    "You are a voice agent speaking on a live phone call. Always reply in "
    "the same language the customer most recently used (English, Hindi, or "
    "Tamil). Keep replies short and natural — 1 to 2 sentences — since this "
    "is spoken aloud, not read. Never use markdown, bullet points, or emoji. "
    "\n\nYour specific instructions for this call, set by the platform "
    "operator, are:\n{operator_prompt}"
)

CREDENTIALS_PATH = Path(os.environ.get("CREDENTIALS_PATH", "data/twilio_credentials.enc"))

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+")

# =============================================================================
# Logging
# =============================================================================

def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
    for noisy in ("httpx", "websockets", "twilio.http_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


settings = Settings.from_env()
configure_logging(settings.log_level)
logger = logging.getLogger("voice_ai_platform")

# =============================================================================
# Twilio credential storage (encrypted, separate from everything else)
# =============================================================================

@dataclass
class TwilioCredentials:
    account_sid: str
    auth_token: str
    from_number: str


class CredentialsStore:
    def __init__(self):
        key = os.environ.get("CREDENTIALS_ENCRYPTION_KEY")
        if not key:
            raise RuntimeError(
                "CREDENTIALS_ENCRYPTION_KEY is not set. Generate one with:\n"
                '  python -c "from cryptography.fernet import Fernet; '
                'print(Fernet.generate_key().decode())"'
            )
        self._fernet = Fernet(key.encode())
        CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)

    def save(self, creds: TwilioCredentials) -> None:
        payload = json.dumps(asdict(creds)).encode()
        CREDENTIALS_PATH.write_bytes(self._fernet.encrypt(payload))

    def load(self) -> TwilioCredentials | None:
        if not CREDENTIALS_PATH.exists():
            return None
        try:
            payload = self._fernet.decrypt(CREDENTIALS_PATH.read_bytes())
            return TwilioCredentials(**json.loads(payload))
        except InvalidToken:
            return None

    def clear(self) -> None:
        CREDENTIALS_PATH.unlink(missing_ok=True)

    def status(self) -> dict:
        creds = self.load()
        if not creds:
            return {"configured": False}
        masked = creds.account_sid[:6] + "…" + creds.account_sid[-4:]
        return {"configured": True, "account_sid": masked, "from_number": creds.from_number}


# =============================================================================
# Call context (operator's prompt + target number, per call)
# =============================================================================

@dataclass
class CallContext:
    call_id: str
    to_number: str
    operator_prompt: str
    twilio_call_sid: str | None = None
    status: str = "initiated"
    transcript: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)


class CallContextStore:
    def __init__(self, ttl_seconds: int):
        self._contexts: dict[str, CallContext] = {}
        self._ttl_seconds = ttl_seconds

    def create(self, to_number: str, operator_prompt: str) -> CallContext:
        call_id = uuid.uuid4().hex[:12]
        ctx = CallContext(call_id=call_id, to_number=to_number, operator_prompt=operator_prompt)
        self._contexts[call_id] = ctx
        return ctx

    def get(self, call_id: str) -> CallContext | None:
        return self._contexts.get(call_id)

    def set_twilio_sid(self, call_id: str, sid: str) -> None:
        if ctx := self._contexts.get(call_id):
            ctx.twilio_call_sid = sid

    def update_status(self, call_id: str, status: str) -> None:
        if ctx := self._contexts.get(call_id):
            ctx.status = status

    def append_transcript(self, call_id: str, role: str, text: str) -> None:
        if ctx := self._contexts.get(call_id):
            ctx.transcript.append({"role": role, "text": text, "t": time.time()})

    def list_recent(self, limit: int = 25) -> list[CallContext]:
        return sorted(self._contexts.values(), key=lambda c: c.created_at, reverse=True)[:limit]

    def sweep_stale(self) -> list[str]:
        now = time.monotonic()
        stale = [cid for cid, c in self._contexts.items() if now - c.created_at > self._ttl_seconds]
        for cid in stale:
            self._contexts.pop(cid, None)
        return stale


# =============================================================================
# Call session (live conversation state during a media stream)
# =============================================================================

@dataclass
class CallSession:
    stream_sid: str
    call_id: str
    history: list[dict]
    last_language: str = "en"
    last_active: float = field(default_factory=time.monotonic)

    @classmethod
    def start(cls, stream_sid: str, call_id: str, system_prompt: str) -> "CallSession":
        return cls(stream_sid=stream_sid, call_id=call_id,
                    history=[{"role": "system", "content": system_prompt}])

    def touch(self) -> None:
        self.last_active = time.monotonic()

    def add_user_turn(self, text: str, max_history_turns: int) -> None:
        self.history.append({"role": "user", "content": text})
        self._trim(max_history_turns)
        self.touch()

    def add_assistant_turn(self, text: str, max_history_turns: int) -> None:
        self.history.append({"role": "assistant", "content": text})
        self._trim(max_history_turns)
        self.touch()

    def _trim(self, max_history_turns: int) -> None:
        system, *rest = self.history
        max_messages = max_history_turns * 2
        if len(rest) > max_messages:
            rest = rest[-max_messages:]
        self.history = [system, *rest]


class SessionStore:
    def __init__(self, ttl_seconds: int):
        self._sessions: dict[str, CallSession] = {}
        self._ttl_seconds = ttl_seconds

    def create(self, stream_sid: str, call_id: str, system_prompt: str) -> CallSession:
        session = CallSession.start(stream_sid, call_id, system_prompt)
        self._sessions[stream_sid] = session
        return session

    def get(self, stream_sid: str) -> CallSession | None:
        return self._sessions.get(stream_sid)

    def remove(self, stream_sid: str) -> None:
        self._sessions.pop(stream_sid, None)

    def sweep_stale(self) -> list[str]:
        now = time.monotonic()
        stale = [sid for sid, s in self._sessions.items() if now - s.last_active > self._ttl_seconds]
        for sid in stale:
            self._sessions.pop(sid, None)
        return stale

    def __len__(self) -> int:
        return len(self._sessions)


# =============================================================================
# STT / LLM / TTS — all via free-tier hosted APIs, no local models
# =============================================================================

class SpeechToText:
    """Groq's hosted Whisper API — no local model, so this runs fine on
    low-RAM free-tier hosting (unlike running faster-whisper locally)."""

    def __init__(self, groq_client: Groq):
        self._client = groq_client

    async def transcribe(self, pcm_audio: bytes, sample_rate: int = 8000) -> tuple[str, str]:
        if not pcm_audio:
            return "", "en"

        segment = AudioSegment(data=pcm_audio, sample_width=2, frame_rate=sample_rate, channels=1)
        wav_io = io.BytesIO()
        segment.export(wav_io, format="wav")
        wav_bytes = wav_io.getvalue()

        try:
            response = await asyncio.to_thread(
                self._client.audio.transcriptions.create,
                file=("turn.wav", wav_bytes),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
            text = (response.text or "").strip()
            language = getattr(response, "language", None) or "en"
            return text, language
        except Exception:
            logger.exception("STT transcription failed")
            return "", "en"


class LLMClient:
    def __init__(self, groq_client: Groq, settings: Settings):
        self._client = groq_client
        self._model = settings.llm_model
        self._max_tokens = settings.llm_max_tokens
        self._max_retries = settings.llm_max_retries
        self._base_delay = settings.llm_retry_base_delay

    async def reply(self, messages: list[dict]) -> str:
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = await asyncio.to_thread(
                    self._client.chat.completions.create,
                    model=self._model,
                    messages=messages,
                    max_tokens=self._max_tokens,
                )
                return response.choices[0].message.content.strip()
            except APIError as e:
                last_error = e
                delay = self._base_delay * (2 ** attempt)
                logger.warning(f"LLM call failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
        logger.error(f"LLM call failed after {self._max_retries} attempts: {last_error}")
        return "Sorry, I'm having trouble responding right now. Could you say that again?"


class TextToSpeech:
    @staticmethod
    def split_into_sentences(text: str) -> list[str]:
        chunks = [c.strip() for c in _SENTENCE_SPLIT.split(text) if c.strip()]
        return chunks or [text]

    async def synthesize_to_mulaw(self, text: str, language: str = "en") -> bytes:
        lang = TTS_LANG_MAP.get(language, DEFAULT_TTS_LANG)
        try:
            mp3_io = await asyncio.to_thread(self._synthesize_sync, text, lang)
        except Exception:
            logger.exception(f"TTS synthesis failed for language={lang}")
            return b""

        if not mp3_io or mp3_io.getbuffer().nbytes == 0:
            return b""
        segment = (
            AudioSegment.from_mp3(mp3_io)
            .set_frame_rate(8000)
            .set_channels(1)
            .set_sample_width(2)  # lin2ulaw assumes 16-bit input; force it explicitly
        )
        return audioop.lin2ulaw(segment.raw_data, 2)

    @staticmethod
    def _synthesize_sync(text: str, lang: str) -> io.BytesIO:
        # gTTS does blocking network IO, so this runs off the event loop via asyncio.to_thread.
        buf = io.BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(buf)
        buf.seek(0)
        return buf


# =============================================================================
# Twilio outbound call trigger
# =============================================================================

class TwilioCallError(Exception):
    pass


class TwilioCallManager:
    def __init__(self, credentials_store: CredentialsStore, public_host: str):
        self._store = credentials_store
        self._public_host = public_host

    def trigger_call(self, to_number: str, call_id: str) -> str:
        creds = self._store.load()
        if not creds:
            raise TwilioCallError("Twilio credentials are not configured yet.")

        client = TwilioClient(creds.account_sid, creds.auth_token)
        voice_url = f"https://{self._public_host}/voice?call_id={call_id}"
        status_url = f"https://{self._public_host}/call-status?call_id={call_id}"

        try:
            call = client.calls.create(
                to=to_number,
                from_=creds.from_number,
                url=voice_url,
                status_callback=status_url,
                status_callback_event=["initiated", "ringing", "answered", "completed"],
            )
            return call.sid
        except Exception as e:
            logger.exception("Failed to trigger Twilio call")
            raise TwilioCallError(str(e)) from e


# =============================================================================
# App wiring
# =============================================================================

app = FastAPI(title="Multilingual Voice AI Platform")

credentials_store = CredentialsStore()
call_contexts = CallContextStore(ttl_seconds=settings.call_context_ttl_seconds)
sessions = SessionStore(ttl_seconds=settings.session_ttl_seconds)
call_manager = TwilioCallManager(credentials_store, settings.public_host)

_groq_client = Groq(api_key=settings.groq_api_key)
stt = SpeechToText(_groq_client)
llm = LLMClient(_groq_client, settings)
tts = TextToSpeech()
vad = webrtcvad.Vad(settings.vad_aggressiveness)

_startup_time = datetime.now(timezone.utc)


@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(_sweeper())


async def _sweeper():
    while True:
        await asyncio.sleep(60)
        stale_sessions = sessions.sweep_stale()
        stale_calls = call_contexts.sweep_stale()
        if stale_sessions or stale_calls:
            logger.info(f"Swept {len(stale_sessions)} session(s), {len(stale_calls)} call context(s)")


# ---- REST API used by the UI -----------------------------------------------

class SaveCredentialsRequest(BaseModel):
    account_sid: str = Field(min_length=1)
    auth_token: str = Field(min_length=1)
    from_number: str = Field(min_length=1)


@app.post("/api/credentials")
async def save_credentials(req: SaveCredentialsRequest):
    credentials_store.save(TwilioCredentials(
        account_sid=req.account_sid.strip(),
        auth_token=req.auth_token.strip(),
        from_number=req.from_number.strip(),
    ))
    logger.info("Twilio credentials updated")
    return credentials_store.status()


@app.get("/api/credentials/status")
async def credentials_status():
    return credentials_store.status()


@app.delete("/api/credentials")
async def delete_credentials():
    credentials_store.clear()
    return {"configured": False}


class TriggerCallRequest(BaseModel):
    to_number: str = Field(min_length=8, description="E.164 format, e.g. +919876543210")
    prompt: str = Field(min_length=1, description="Instructions for what the agent should do on this call")


@app.post("/api/calls")
async def trigger_call(req: TriggerCallRequest):
    if not credentials_store.status()["configured"]:
        raise HTTPException(400, "Save Twilio credentials before starting a call.")

    ctx = call_contexts.create(to_number=req.to_number.strip(), operator_prompt=req.prompt.strip())
    try:
        call_sid = call_manager.trigger_call(ctx.to_number, ctx.call_id)
    except TwilioCallError as e:
        call_contexts.update_status(ctx.call_id, "failed")
        raise HTTPException(502, f"Twilio call failed: {e}")

    call_contexts.set_twilio_sid(ctx.call_id, call_sid)
    logger.info(f"Triggered call {ctx.call_id} (twilio_sid={call_sid}) to {ctx.to_number}")
    return {"call_id": ctx.call_id, "twilio_call_sid": call_sid, "status": ctx.status}


@app.get("/api/calls")
async def list_calls():
    return [
        {
            "call_id": c.call_id,
            "to_number": c.to_number,
            "status": c.status,
            "twilio_call_sid": c.twilio_call_sid,
            "prompt_preview": c.operator_prompt[:80],
            "transcript": c.transcript,
        }
        for c in call_contexts.list_recent()
    ]


@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ok",
        "uptime_seconds": (datetime.now(timezone.utc) - _startup_time).total_seconds(),
        "active_sessions": len(sessions),
        "twilio_configured": credentials_store.status()["configured"],
    })


# ---- Twilio webhooks --------------------------------------------------------

@app.post("/voice")
async def voice(request: Request, call_id: str):
    ctx = call_contexts.get(call_id)
    if not ctx:
        return Response(
            content='<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>',
            media_type="application/xml",
        )
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{settings.public_host}/media">
      <Parameter name="call_id" value="{call_id}" />
    </Stream>
  </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


@app.post("/call-status")
async def call_status(request: Request, call_id: str):
    form = await request.form()
    status = form.get("CallStatus", "unknown")
    call_contexts.update_status(call_id, status)
    logger.info(f"[{call_id}] status -> {status}")
    return Response(status_code=204)


# ---- Media stream: the live audio pipeline ---------------------------------

@app.websocket("/media")
async def media(ws: WebSocket):
    await ws.accept()

    stream_sid: str | None = None
    call_id: str | None = None
    ctx = None
    audio_buffer = bytearray()
    silence_ms = 0
    speaking = False
    frame_ms = 20

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "start":
                stream_sid = msg["start"]["streamSid"]
                custom_params = msg["start"].get("customParameters", {})
                call_id = custom_params.get("call_id")

                if not call_id:
                    logger.warning("Media stream 'start' event had no call_id parameter")
                    await ws.close(code=1008)
                    return

                ctx = call_contexts.get(call_id)
                if not ctx:
                    logger.warning(f"[{call_id}] no call context found for media stream")
                    await ws.close(code=1008)
                    return

                system_prompt = BASE_SYSTEM_INSTRUCTIONS.format(operator_prompt=ctx.operator_prompt)
                sessions.create(stream_sid, call_id, system_prompt)
                call_contexts.update_status(call_id, "in-progress")
                logger.info(f"[{call_id}] media stream started")
                asyncio.create_task(_greet(ws, stream_sid, call_id))

            elif event == "media" and stream_sid:
                mulaw_chunk = base64.b64decode(msg["media"]["payload"])
                pcm_chunk = audioop.ulaw2lin(mulaw_chunk, 2)
                try:
                    is_speech = vad.is_speech(pcm_chunk[:320], sample_rate=8000)
                except Exception:
                    is_speech = False

                if is_speech:
                    audio_buffer.extend(pcm_chunk)
                    speaking = True
                    silence_ms = 0
                elif speaking:
                    silence_ms += frame_ms
                    audio_buffer.extend(pcm_chunk)
                    if silence_ms >= settings.silence_ms_to_end_turn:
                        turn_audio = bytes(audio_buffer)
                        audio_buffer = bytearray()
                        speaking = False
                        silence_ms = 0
                        asyncio.create_task(_handle_turn(ws, stream_sid, call_id, turn_audio))

            elif event == "stop":
                logger.info(f"[{call_id}] media stream stopped")
                call_contexts.update_status(call_id, "completed")
                if stream_sid:
                    sessions.remove(stream_sid)
                break

    except WebSocketDisconnect:
        if stream_sid:
            sessions.remove(stream_sid)
        call_contexts.update_status(call_id, "completed")
    except Exception:
        logger.exception(f"[{call_id}] unexpected error in media stream")
        if stream_sid:
            sessions.remove(stream_sid)
        call_contexts.update_status(call_id, "failed")
        with contextlib.suppress(Exception):
            await ws.close()


async def _greet(ws: WebSocket, stream_sid: str, call_id: str):
    session = sessions.get(stream_sid)
    if session is None:
        return
    session.history.append({
        "role": "user",
        "content": "The call has just connected. Greet the customer and open the conversation per your instructions.",
    })
    reply = await llm.reply(session.history)
    session.add_assistant_turn(reply, settings.max_history_turns)
    call_contexts.append_transcript(call_id, "agent", reply)
    for sentence in tts.split_into_sentences(reply):
        audio = await tts.synthesize_to_mulaw(sentence, language=session.last_language)
        if audio:
            await _stream_audio(ws, stream_sid, audio)


async def _handle_turn(ws: WebSocket, stream_sid: str, call_id: str, pcm_audio: bytes):
    session = sessions.get(stream_sid)
    if session is None:
        return

    t0 = time.monotonic()
    text, language = await stt.transcribe(pcm_audio)
    if not text:
        return
    session.last_language = language
    logger.info(f"[{call_id}] customer ({language}): {text}  [{time.monotonic() - t0:.2f}s]")
    call_contexts.append_transcript(call_id, "customer", text)

    session.add_user_turn(text, settings.max_history_turns)

    t1 = time.monotonic()
    reply = await llm.reply(session.history)
    logger.info(f"[{call_id}] agent: {reply}  [{time.monotonic() - t1:.2f}s]")
    session.add_assistant_turn(reply, settings.max_history_turns)
    call_contexts.append_transcript(call_id, "agent", reply)

    for sentence in tts.split_into_sentences(reply):
        if sessions.get(stream_sid) is None:
            return
        audio = await tts.synthesize_to_mulaw(sentence, language=language)
        if audio:
            await _stream_audio(ws, stream_sid, audio)


async def _stream_audio(ws: WebSocket, stream_sid: str, mulaw_audio: bytes):
    chunk_size = 160
    try:
        for i in range(0, len(mulaw_audio), chunk_size):
            chunk = mulaw_audio[i : i + chunk_size]
            payload = base64.b64encode(chunk).decode("ascii")
            await ws.send_text(json.dumps({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": payload},
            }))
            await asyncio.sleep(0.02)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception(f"[{stream_sid}] failed to stream audio back")


# =============================================================================
# Frontend — embedded so the whole app is one file
# =============================================================================

FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Voice AI Platform</title>
<style>
  :root { --bg:#f6f5f2; --card:#fff; --border:#ddd9d0; --text:#24231f; --text-muted:#726f66; --accent:#3c3489; --accent-hover:#26215c; font-family:-apple-system,"Segoe UI",Roboto,sans-serif; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); }
  main { max-width:720px; margin:0 auto; padding:32px 20px 80px; }
  h1 { font-size:24px; font-weight:500; margin-bottom:4px; }
  h2 { font-size:16px; font-weight:500; margin:0 0 12px; }
  .subtitle { color:var(--text-muted); margin-top:0; margin-bottom:28px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px; margin-bottom:20px; }
  label { display:block; font-size:13px; color:var(--text-muted); margin-bottom:12px; }
  input, textarea { display:block; width:100%; margin-top:6px; padding:9px 10px; border:1px solid var(--border); border-radius:8px; font-size:14px; font-family:inherit; color:var(--text); background:var(--bg); }
  textarea { resize:vertical; }
  button { background:var(--accent); color:#fff; border:none; border-radius:8px; padding:10px 16px; font-size:14px; cursor:pointer; }
  button:hover { background:var(--accent-hover); }
  button.secondary { background:transparent; color:var(--accent); border:1px solid var(--border); margin-bottom:12px; }
  button.secondary:hover { background:var(--bg); }
  .hint { font-size:13px; color:var(--text-muted); margin-top:10px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:8px; border-bottom:1px solid var(--border); vertical-align:top; }
  th { color:var(--text-muted); font-weight:500; }
</style>
</head>
<body>
  <main>
    <h1>Multilingual voice AI platform</h1>
    <p class="subtitle">Trigger outbound calls to customers and let the agent handle the conversation in English, Hindi, or Tamil.</p>

    <section class="card">
      <h2>1. Twilio credentials</h2>
      <p class="hint" id="cred-status">Checking status…</p>
      <label>Account SID <input id="account_sid" type="text" placeholder="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" /></label>
      <label>Auth token <input id="auth_token" type="password" placeholder="Your Twilio auth token" /></label>
      <label>From number <input id="from_number" type="text" placeholder="+1XXXXXXXXXX" /></label>
      <button id="save-creds-btn">Save credentials</button>
    </section>

    <section class="card">
      <h2>2. Start a call</h2>
      <label>Customer mobile number <input id="to_number" type="text" placeholder="+91XXXXXXXXXX" /></label>
      <label>Agent instructions
        <textarea id="prompt" rows="5" placeholder="e.g. You are calling on behalf of Acme Bank to confirm the customer's appointment tomorrow at 3pm."></textarea>
      </label>
      <button id="start-call-btn">Start call</button>
      <p class="hint" id="call-trigger-status"></p>
    </section>

    <section class="card">
      <h2>3. Recent calls</h2>
      <button id="refresh-btn" class="secondary">Refresh</button>
      <table id="calls-table">
        <thead><tr><th>To</th><th>Status</th><th>Prompt</th><th>Transcript</th></tr></thead>
        <tbody></tbody>
      </table>
    </section>
  </main>

<script>
async function refreshCredentialStatus() {
  const res = await fetch("/api/credentials/status");
  const data = await res.json();
  const el = document.getElementById("cred-status");
  el.textContent = data.configured
    ? `Configured — ${data.account_sid}, calling from ${data.from_number}`
    : "Not configured yet — enter your Twilio credentials below.";
}

async function saveCredentials() {
  const account_sid = document.getElementById("account_sid").value.trim();
  const auth_token = document.getElementById("auth_token").value.trim();
  const from_number = document.getElementById("from_number").value.trim();
  if (!account_sid || !auth_token || !from_number) { alert("Fill in all three Twilio fields."); return; }

  const res = await fetch("/api/credentials", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ account_sid, auth_token, from_number }),
  });
  if (res.ok) {
    document.getElementById("account_sid").value = "";
    document.getElementById("auth_token").value = "";
    document.getElementById("from_number").value = "";
    await refreshCredentialStatus();
  } else {
    alert("Failed to save credentials. Check the server logs.");
  }
}

async function startCall() {
  const to_number = document.getElementById("to_number").value.trim();
  const prompt = document.getElementById("prompt").value.trim();
  const status = document.getElementById("call-trigger-status");
  if (!to_number || !prompt) { alert("Enter both a phone number and agent instructions."); return; }

  status.textContent = "Starting call…";
  const res = await fetch("/api/calls", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ to_number, prompt }),
  });
  const data = await res.json();
  if (res.ok) {
    status.textContent = `Call started (id: ${data.call_id}). Status: ${data.status}`;
    await refreshCalls();
  } else {
    status.textContent = `Failed: ${data.detail || "unknown error"}`;
  }
}

async function refreshCalls() {
  const res = await fetch("/api/calls");
  const calls = await res.json();
  const tbody = document.querySelector("#calls-table tbody");
  tbody.innerHTML = "";
  for (const call of calls) {
    const tr = document.createElement("tr");
    const transcriptText = call.transcript.map(t => `${t.role}: ${t.text}`).join(" | ") || "—";
    tr.innerHTML = `<td>${escapeHtml(call.to_number)}</td><td>${escapeHtml(call.status)}</td><td>${escapeHtml(call.prompt_preview)}</td><td>${escapeHtml(transcriptText)}</td>`;
    tbody.appendChild(tr);
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

document.getElementById("save-creds-btn").addEventListener("click", saveCredentials);
document.getElementById("start-call-btn").addEventListener("click", startCall);
document.getElementById("refresh-btn").addEventListener("click", refreshCalls);

refreshCredentialStatus();
refreshCalls();
setInterval(refreshCalls, 5000);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def frontend():
    return FRONTEND_HTML
