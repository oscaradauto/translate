"""Configuración de Stage 1 - Meeting Subtitles."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

MEETING_LANGUAGE = "en"

# Micrófono físico. Este índice corresponde al índice usado por PyAudio.
MIC_DEVICE_INDEX = int(os.getenv("MIC_DEVICE_INDEX", "1"))

# Motor especializado de streaming STT.
OPENAI_TRANSCRIPTION_MODEL = os.getenv(
    "OPENAI_TRANSCRIPTION_MODEL",
    "gpt-live-transcribe",
)
OPENAI_TRANSCRIPTION_DELAY = os.getenv(
    "OPENAI_TRANSCRIPTION_DELAY",
    "low",
)

# Contexto para mejorar nombres propios y vocabulario de la reunión técnica.
OPENAI_TRANSCRIPTION_PROMPT = os.getenv(
    "OPENAI_TRANSCRIPTION_PROMPT",
    (
        "English software engineering team meeting. "
        "Expect technical discussions about Java, Spring Boot, Kafka, Azure, "
        "Azure Functions, GraphQL, REST APIs, GitHub, Jira, JQL, MXL, PDP, PLP, "
        "APAC, Q3, Q4, Q1, epics, dependencies, latency, cold starts, "
        "circuit breakers, timeouts, performance testing, load testing, "
        "mobile applications, schemas, tickets, pull requests, and deployments. "
        "Preserve technical terms, acronyms, names, ticket numbers, and product "
        "names as spoken."
    ),
)

OPENAI_TRANSCRIPTION_KEYWORDS = [
    value.strip()
    for value in os.getenv(
        "OPENAI_TRANSCRIPTION_KEYWORDS",
        (
            "MXL, PDP, PLP, JQL, GraphQL, REST API, API, endpoint, "
            "Azure Functions, Azure Function Apps, GitHub Actions, Kafka, "
            "Spring Boot, Spring Cloud, Java, Jira, AEM, schema, spike ticket, "
            "epic, dependency map, quarterly planning, Q4, Q1, Q3, APAC, "
            "mobile team, cold start, latency, circuit breaker, timeout, "
            "read timeout, cart endpoint, getCart, production, dev URL, "
            "performance testing, load testing, Store Mode, user info, "
            "observability, tracing, pull request, dependency map, "
            "Jared, Hyago, Kaushik, Ananda, George, Brittany, Sharada, Manas"
        ),
    ).split(",")
    if value.strip()
]


def get_meeting_language():
    return MEETING_LANGUAGE
