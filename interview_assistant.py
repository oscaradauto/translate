"""OpenAI answer engine for Stage 2 technical interviews.

Stage 2 intentionally keeps transcription local. This module only sends the
resolved technical question plus a short recent conversation window to OpenAI.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Iterable

from openai import OpenAI

from config import (
    ASSISTANT_MAX_OUTPUT_TOKENS,
    ASSISTANT_REASONING_EFFORT,
    OPENAI_ASSISTANT_MODEL,
)

DeltaCallback = Callable[[str], None]


SYSTEM_PROMPT_EN = """You are a senior software engineer helping during a live technical interview.
Answer in natural spoken English. Be direct, technically precise, concise, and easy to say aloud.
Do not start with filler such as 'Great question', 'Sure', or 'Absolutely'.
Prefer 2-4 short sentences for conceptual questions.
For architecture/design questions, give the main approach, the key trade-off, and one practical detail.
For coding questions, keep code very small unless the interviewer explicitly asks for a full implementation.
Use the supplied conversation context to resolve follow-up questions and pronouns.
Do not claim personal experience, employers, incidents, metrics, or projects that were not supplied in the conversation.
If the transcription appears to contain a minor technical-term error, infer the most plausible software term from context without discussing the transcription error.
Return only the answer the interviewee could say aloud."""

SYSTEM_PROMPT_ES = """Eres un desarrollador senior ayudando durante una entrevista técnica en vivo.
Responde en español natural, hablado, directo, técnicamente preciso, breve y fácil de decir en voz alta.
No empieces con relleno como 'Buena pregunta', 'Claro' o 'Por supuesto'.
Para preguntas conceptuales usa normalmente 2-4 frases cortas.
Para arquitectura o diseño da el enfoque principal, el trade-off clave y un detalle práctico.
Para preguntas de código, mantén el código muy corto salvo que el entrevistador pida explícitamente una implementación completa.
Usa el contexto de conversación para resolver preguntas de seguimiento y pronombres.
No inventes experiencia personal, empleadores, incidentes, métricas ni proyectos que no aparezcan en el contexto.
Si la transcripción parece contener un pequeño error en un término técnico, infiere el término de software más probable por contexto sin hablar del error de transcripción.
Devuelve únicamente la respuesta que el entrevistado podría decir en voz alta."""


QUESTION_STARTERS = (
    "what", "how", "why", "when", "where", "which", "who",
    "can you", "could you", "would you", "do you", "did you",
    "have you", "are you", "is there", "are there", "should",
    "tell me", "explain", "walk me through", "describe",
    "what's", "whats", "how's", "hows",
)


def looks_like_question(text: str) -> bool:
    """Fast local filter so ordinary interview chatter does not call OpenAI."""
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return False

    lower = cleaned.casefold()

    if "?" in cleaned:
        return True

    if any(lower.startswith(prefix) for prefix in QUESTION_STARTERS):
        return True

    # Common interview-style requests that may be transcribed without a
    # question mark.
    patterns = (
        r"\bdifference between\b",
        r"\bcompare\b",
        r"\bpros and cons\b",
        r"\btrade[- ]?offs?\b",
        r"\bwhen would you\b",
        r"\bwhat would you\b",
        r"\bhow would you\b",
        r"\bwhy would you\b",
    )
    return any(re.search(pattern, lower) for pattern in patterns)


@dataclass(frozen=True)
class ConversationTurn:
    speaker: str
    text: str


class OpenAIInterviewAssistant:
    """Small, synchronous OpenAI Responses API wrapper."""

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY no está configurada en el archivo .env."
            )

        self.client = OpenAI(
            api_key=api_key,
            timeout=30.0,
            max_retries=1,
        )
        self.model = OPENAI_ASSISTANT_MODEL

    def stream_answer(
        self,
        question: str,
        recent_turns: Iterable[ConversationTurn],
        language: str,
        on_delta: DeltaCallback,
    ) -> str:
        instructions = (
            SYSTEM_PROMPT_ES
            if language == "es"
            else SYSTEM_PROMPT_EN
        )

        context_lines = []
        for turn in recent_turns:
            label = "Interviewer" if turn.speaker == "INTERVIEWER" else "You"
            context_lines.append(f"{label}: {turn.text}")

        context = "\n".join(context_lines[-12:])
        prompt = (
            "Recent interview conversation:\n"
            f"{context}\n\n"
            "Latest interviewer question:\n"
            f"{question}\n\n"
            "Answer the latest question using the recent conversation only as context."
        )

        collected: list[str] = []

        # Responses streaming keeps perceived latency low: text can appear in
        # the UI as soon as the first output token is available.
        try:
            with self.client.responses.stream(
                model=self.model,
                instructions=instructions,
                input=prompt,
                reasoning={"effort": ASSISTANT_REASONING_EFFORT},
                max_output_tokens=ASSISTANT_MAX_OUTPUT_TOKENS,
            ) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        delta = event.delta or ""
                        if delta:
                            collected.append(delta)
                            on_delta(delta)

                stream.get_final_response()

            return "".join(collected).strip()
        except Exception:
            # Compatibility fallback for SDKs/environments where streaming is
            # unavailable. The UI still receives one complete delta.
            response = self.client.responses.create(
                model=self.model,
                instructions=instructions,
                input=prompt,
                reasoning={"effort": ASSISTANT_REASONING_EFFORT},
                max_output_tokens=ASSISTANT_MAX_OUTPUT_TOKENS,
            )
            text = (response.output_text or "").strip()
            if text:
                on_delta(text)
            return text
