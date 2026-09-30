"""Application configuration.

Stage 1 uses Faster-Whisper locally and is always English.
Stage 2 has its own response-language configuration and will use OpenAI.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Stage 1 - Live subtitles
# ---------------------------------------------------------------------------

MEETING_LANGUAGE = "en"

# PyAudio input device index for the physical microphone.
MIC_DEVICE_INDEX = int(os.getenv("MIC_DEVICE_INDEX", "-1"))

# Faster-Whisper model configuration.
# base.en is the default because Stage 1 prioritizes low-latency English captions.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "4"))
WHISPER_NUM_WORKERS = int(os.getenv("WHISPER_NUM_WORKERS", "1"))

# Caption/VAD tuning.
SUBTITLE_PARTIAL_INTERVAL_SECONDS = float(
    os.getenv("SUBTITLE_PARTIAL_INTERVAL_SECONDS", "0.45")
)
SUBTITLE_PARTIAL_MIN_SECONDS = float(
    os.getenv("SUBTITLE_PARTIAL_MIN_SECONDS", "0.9")
)
WHISPER_PARTIAL_WINDOW_SECONDS = float(
    os.getenv("WHISPER_PARTIAL_WINDOW_SECONDS", "4.0")
)
SUBTITLE_SPEECH_END_MS = int(os.getenv("SUBTITLE_SPEECH_END_MS", "700"))
SUBTITLE_MAX_UTTERANCE_SECONDS = float(
    os.getenv("SUBTITLE_MAX_UTTERANCE_SECONDS", "8")
)

# False-positive protection. The physical microphone is intentionally more
# conservative than system audio because speaker bleed / room noise can make
# Whisper hallucinate short words even when the user did not speak.
MIC_VAD_MODE = int(os.getenv("MIC_VAD_MODE", "3"))
MIC_MIN_DBFS = float(os.getenv("MIC_MIN_DBFS", "-40"))
MIC_SPEECH_START_MS = int(os.getenv("MIC_SPEECH_START_MS", "120"))
MIC_MIN_VOICED_MS = int(os.getenv("MIC_MIN_VOICED_MS", "180"))

MEETING_VAD_MODE = int(os.getenv("MEETING_VAD_MODE", "2"))
MEETING_SPEECH_START_MS = int(
    os.getenv("MEETING_SPEECH_START_MS", "120")
)
MEETING_MIN_VOICED_MS = int(
    os.getenv("MEETING_MIN_VOICED_MS", "240")
)

# When local microphone speech is detected, prefer YOU over the system
# loopback for a short hold window. This prevents the user's own voice from
# being emitted as MEETING when Teams/Windows routes a local sidetone or mix
# into the playback capture.
LOCAL_SPEECH_GATE_HOLD_MS = int(
    os.getenv("LOCAL_SPEECH_GATE_HOLD_MS", "800")
)

# Bias the local model toward the vocabulary used in engineering meetings.
WHISPER_INITIAL_PROMPT = os.getenv(
    "WHISPER_INITIAL_PROMPT",
    (
        "English software engineering team meeting. "
        "Technical vocabulary may include Java, Spring Boot, Kafka, Azure, "
        "Azure Functions, GraphQL, REST APIs, GitHub, Jira, JQL, MXL, PDP, PLP, "
        "AEM, APAC, Q1, Q3, Q4, pull request, deployment, latency, cold start, "
        "circuit breaker, timeout, performance testing, load testing, schema, "
        "endpoint, getCart, observability and tracing."
    ),
)


# ---------------------------------------------------------------------------
# Stage 2 - Technical interview assistant
# ---------------------------------------------------------------------------

ASSISTANT_LANGUAGE = os.getenv("ASSISTANT_LANGUAGE", "en")
AI_PROVIDER = os.getenv("AI_PROVIDER", "groq")

# Stage 2 uses a multilingual Faster-Whisper model. "auto" means language is
# detected independently for every utterance, so English and Spanish can be
# mixed during the same interview.
INTERVIEW_WHISPER_MODEL = os.getenv(
    "INTERVIEW_WHISPER_MODEL",
    "base",
)
INTERVIEW_TRANSCRIPTION_LANGUAGE = os.getenv(
    "INTERVIEW_TRANSCRIPTION_LANGUAGE",
    "auto",
).strip().lower()
INTERVIEW_WHISPER_INITIAL_PROMPT = os.getenv(
    "INTERVIEW_WHISPER_INITIAL_PROMPT",
    (
        "Software engineering technical interview in English or Spanish. "
        "Technical vocabulary may include Java, Spring Boot, dependency "
        "injection, Kafka, microservices, REST APIs, GraphQL, SQL, NoSQL, "
        "MongoDB, Azure, AWS, Docker, Kubernetes, CI/CD, SOLID, design "
        "patterns, concurrency, threads, testing and system design."
    ),
)

# Groq is used during development for technical reasoning. Transcription stays
# local with Faster-Whisper, so Groq only receives text/context.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_REASONING_EFFORT = os.getenv(
    "GROQ_REASONING_EFFORT",
    "medium",
)
GROQ_MAX_COMPLETION_TOKENS = int(
    os.getenv("GROQ_MAX_COMPLETION_TOKENS", "420")
)

# Keep enough recent turns to resolve follow-up questions such as
# "and what happens if it fails?" without sending the full interview.
INTERVIEW_CONTEXT_TURNS = int(
    os.getenv("INTERVIEW_CONTEXT_TURNS", "10")
)

# After the interviewer stops speaking, wait briefly before deciding whether
# the turn contains a complete question.
INTERVIEW_QUESTION_DEBOUNCE_SECONDS = float(
    os.getenv("INTERVIEW_QUESTION_DEBOUNCE_SECONDS", "1.2")
)


def get_meeting_language() -> str:
    return MEETING_LANGUAGE


def get_language_mode() -> str:
    """Compatibility helper for legacy assistant code."""
    return ASSISTANT_LANGUAGE if ASSISTANT_LANGUAGE in {"en", "es"} else "en"


def get_ai_provider_name() -> str:
    """Compatibility helper for legacy provider code."""
    return AI_PROVIDER
