"""Application configuration.

Stage 1 uses English-only Faster-Whisper locally.
Stage 2 uses multilingual Faster-Whisper plus Groq during development.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Stage 1 - Live subtitles
# ---------------------------------------------------------------------------


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
# Generic defaults kept for shared capture components. Stage 1 overrides them
# per source below; Stage 2 already supplies its own interview-specific values.
SUBTITLE_SPEECH_END_MS = int(os.getenv("SUBTITLE_SPEECH_END_MS", "700"))
SUBTITLE_MAX_UTTERANCE_SECONDS = float(
    os.getenv("SUBTITLE_MAX_UTTERANCE_SECONDS", "8")
)

# Stage 1 source-specific segmentation.
# YOU gets a slightly longer silence tolerance because natural pauses while
# speaking English were being split into many tiny captions. MEETING keeps a
# tighter silence boundary so remote turn changes are still separated when a
# real pause exists. Longer hard caps reduce arbitrary 8-second sentence cuts
# while rolling partial captions remain available in real time.
MIC_SUBTITLE_SPEECH_END_MS = int(
    os.getenv("MIC_SUBTITLE_SPEECH_END_MS", "1200")
)
MEETING_SUBTITLE_SPEECH_END_MS = int(
    os.getenv("MEETING_SUBTITLE_SPEECH_END_MS", "700")
)
MIC_SUBTITLE_MAX_UTTERANCE_SECONDS = float(
    os.getenv("MIC_SUBTITLE_MAX_UTTERANCE_SECONDS", "14")
)
MEETING_SUBTITLE_MAX_UTTERANCE_SECONDS = float(
    os.getenv("MEETING_SUBTITLE_MAX_UTTERANCE_SECONDS", "18")
)

# Keep live partial captions fast, but let the final pass spend a little more
# search effort for technical terms once the utterance is complete.
STAGE1_WHISPER_FINAL_BEAM_SIZE = int(
    os.getenv("STAGE1_WHISPER_FINAL_BEAM_SIZE", "3")
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
        "Technical vocabulary may include ADR, ADRs, Architecture Decision "
        "Record, architecture decision records, sensible defaults, Java, "
        "Spring Boot, Kafka, Azure, Azure Functions, GraphQL, REST APIs, SDK, "
        "SDKs, API, APIs, GitHub, pull request, PR, repository, repositories, "
        "Jira, JQL, MXL, PDP, PLP, AEM, APAC, iOS, Android, DRY principle, "
        "Tailwind, Tailwind CSS, Sass, SCSS, Lottie, SVG, Postman, Proxyman, "
        "ServiceNow, HAR logs, CTASK, Confluence, MCP, serverless, coding agents, "
        "Q1, Q3, Q4, deployment, latency, cold start, circuit breaker, circuit "
        "breakers, acceptance criteria, story points, schema PR, refinement, "
        "Kanban, OpenTelemetry, New Relic, Application Insights, App Insights, "
        "ContentSquare, Klarna, Afterpay, Azure Function Apps, virtual threads, "
        "Java 21, application-level executor, shared executor, feature flag, "
        "trace ID, operation key, dependency key, downstream service, telemetry, "
        "instrumentation, P50, P95, P99, conversion rate, checkout completion, "
        "access token, refresh token, high-priority, priority, Jira links, "
        "ticket links, blocked by, depends on, code freeze, mobile regression, "
        "production release, cart API, wallet info API, dev environment, "
        "live traffic, release regression, timeout, cache validation, state "
        "management, memory management, garbage collection, authentication, "
        "OAuth, performance testing, load testing, QA, schema, endpoint, getCart, "
        "observability and tracing."
    ),
)


# ---------------------------------------------------------------------------
# Stage 2 - Technical interview assistant
# ---------------------------------------------------------------------------

ASSISTANT_LANGUAGE = os.getenv("ASSISTANT_LANGUAGE", "en")

# Which speaker is allowed to trigger Stage 2 answers.
# interviewer: real interview mode (recommended)
# both: useful for solo testing or interactive assistant usage
ASSISTANT_RESPONSE_SCOPE = os.getenv(
    "ASSISTANT_RESPONSE_SCOPE",
    "interviewer",
).strip().lower()

# Stage 2 prioritizes transcription quality over local-only execution.
# Groq Whisper Large V3 handles English/Spanish input; the local multilingual
# Faster-Whisper model remains available as an offline fallback.
INTERVIEW_STT_PROVIDER = os.getenv(
    "INTERVIEW_STT_PROVIDER",
    "groq",
).strip().lower()
INTERVIEW_WHISPER_MODEL = os.getenv(
    "INTERVIEW_WHISPER_MODEL",
    "small",
)
INTERVIEW_TRANSCRIPTION_LANGUAGE = os.getenv(
    "INTERVIEW_TRANSCRIPTION_LANGUAGE",
    "auto",
).strip().lower()
GROQ_STT_MODEL = os.getenv(
    "GROQ_STT_MODEL",
    "whisper-large-v3",
)
GROQ_STT_PROMPT = os.getenv(
    "GROQ_STT_PROMPT",
    (
        "Technical software interview in English or Spanish. Questions may "
        "mix Spanish with English programming terminology. Java, JVM, JPA, "
        "Spring Boot, dependency injection, @Autowired, Autowired, JWT, OAuth, "
        "OAuth2, OIDC, authentication, authorization, HTTP, REST, GET, POST, "
        "PUT, PATCH, DELETE, Spring MVC, RestController, GetMapping, "
        "PostMapping, PutMapping, PatchMapping, DeleteMapping, Circuit Breaker, "
        "Resilience4j, Kafka, microservices, GraphQL, SQL, NoSQL, MongoDB, "
        "Redis, Azure, AWS, Docker, Kubernetes, CI/CD, SOLID, Java Streams, "
        "Stream, Collectors, distinct, ArrayList, List, List<Integer>, "
        "Collection, Set, HashSet, HashMap, int[], Integer[], array, anagram, "
        "isAnagram, getDuplicate, duplicates, duplicate elements, LocalDate, "
        "Period, heap, stack, garbage collection."
    ),
)

# Stage 2 language-aware Whisper prompts. The base vocabulary above remains
# reusable and can still be overridden with GROQ_STT_PROMPT. These wrappers
# describe how natural-language grammar and English technical vocabulary mix
# in real software-engineering interviews.
GROQ_STT_PROMPT_AUTO = os.getenv(
    "GROQ_STT_PROMPT_AUTO",
    (
        "Software-engineering interview. The speaker may use English, Spanish, "
        "or Spanish sentences containing English technical terminology. "
        "Preserve established technical terms and acronyms in their canonical "
        "English spelling when clearly spoken. Do not convert the surrounding "
        "sentence to another language merely because it contains English "
        "technical terms. "
        + GROQ_STT_PROMPT
    ),
)

GROQ_STT_PROMPT_EN = os.getenv(
    "GROQ_STT_PROMPT_EN",
    (
        "The primary spoken language is English. This is a technical software "
        "engineering interview. Transcribe the sentence in English and preserve "
        "canonical technical spelling for terms such as AWS, Azure, Java, "
        "Spring Boot, @Autowired, @Repository, JWT, OAuth, Circuit Breaker, "
        "Garbage Collector, Lambda, HashMap, ConcurrentHashMap, Kafka, GraphQL, "
        "REST API, Docker, Kubernetes, CI/CD, Java Streams, and Optional. "
        "Do not phonetically rewrite established technical terms. "
        + GROQ_STT_PROMPT
    ),
)

GROQ_STT_PROMPT_ES = os.getenv(
    "GROQ_STT_PROMPT_ES",
    (
        "El idioma principal hablado es español. Esta es una entrevista técnica "
        "de software. La gramática y las palabras funcionales de la oración son "
        "españolas, pero los términos técnicos se pronuncian con frecuencia en "
        "inglés. Mantén esos términos con su escritura técnica canónica en inglés "
        "y no cambies la oración completa a inglés, italiano u otro idioma solo "
        "por escucharlos. Ejemplos de términos que deben conservarse: AWS, Azure, "
        "Java, Spring Boot, @Autowired, @Repository, JWT, OAuth, Circuit Breaker, "
        "Garbage Collector, Lambda, HashMap, ConcurrentHashMap, Kafka, GraphQL, "
        "REST API, Docker, Kubernetes, CI/CD, Java Streams y Optional. "
        "No traduzcas ni reescribas fonéticamente términos técnicos establecidos. "
        + GROQ_STT_PROMPT
    ),
)
INTERVIEW_WHISPER_INITIAL_PROMPT = os.getenv(
    "INTERVIEW_WHISPER_INITIAL_PROMPT",
    GROQ_STT_PROMPT,
)

# Interview questions should not be split using the aggressive 8-second
# subtitle limit. Finalize primarily on a natural pause.
INTERVIEW_SPEECH_END_MS = int(
    os.getenv("INTERVIEW_SPEECH_END_MS", "900")
)
INTERVIEW_MAX_UTTERANCE_SECONDS = float(
    os.getenv("INTERVIEW_MAX_UTTERANCE_SECONDS", "30")
)

# Groq is used during development for both high-accuracy Stage 2 speech
# recognition and technical reasoning. Stage 1 remains fully local.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_REASONING_EFFORT = os.getenv(
    "GROQ_REASONING_EFFORT",
    "medium",
)
GROQ_MAX_COMPLETION_TOKENS = int(
    os.getenv("GROQ_MAX_COMPLETION_TOKENS", "260")
)
GROQ_CONTINUATION_MAX_COMPLETION_TOKENS = int(
    os.getenv("GROQ_CONTINUATION_MAX_COMPLETION_TOKENS", "180")
)

# Coding / whiteboarding answers may need a short explanation plus a complete
# implementation. Keep this separate from normal technical answers so regular
# interview responses remain concise.
GROQ_CODING_MAX_COMPLETION_TOKENS = int(
    os.getenv("GROQ_CODING_MAX_COMPLETION_TOKENS", "650")
)

# Default implementation language for coding follow-ups such as
# "implement it" or "do the same with streams".
INTERVIEW_CODE_LANGUAGE = os.getenv(
    "INTERVIEW_CODE_LANGUAGE",
    "java",
).strip().lower()

# Keep enough recent turns to resolve follow-up questions such as
# "and what happens if it fails?" without sending the full interview.
INTERVIEW_CONTEXT_TURNS = int(
    os.getenv("INTERVIEW_CONTEXT_TURNS", "24")
)

# After the interviewer stops speaking, wait briefly before deciding whether
# the turn contains a complete question.
INTERVIEW_QUESTION_DEBOUNCE_SECONDS = float(
    os.getenv("INTERVIEW_QUESTION_DEBOUNCE_SECONDS", "1.6")
)
