"""Groq answer engine for Stage 2 technical interviews.

Transcription remains local with Faster-Whisper. Only text from the recent
interview context is sent to Groq when a candidate interviewer turn is ready.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable

from groq import Groq

from config import (
    GROQ_MAX_COMPLETION_TOKENS,
    GROQ_MODEL,
    GROQ_REASONING_EFFORT,
)

DeltaCallback = Callable[[str], None]


SYSTEM_PROMPT_EN = """You are a senior software engineer helping during a live technical interview.
Answer in natural spoken English. Be direct, technically precise, concise, and easy to say aloud.
Do not start with filler such as 'Great question', 'Sure', or 'Absolutely'.
Prefer 2-4 short sentences for conceptual questions.
For architecture or system-design questions, give the main approach, the key trade-off, and one practical detail.
For coding questions, explain the approach first and keep code very small unless the interviewer explicitly asks for a full implementation.
Use the recent conversation context to resolve follow-up questions and pronouns.
Do not invent personal experience, employers, incidents, metrics, or projects.
If the transcription contains a likely minor technical-term error, infer the most plausible software term from context without discussing the transcription error.
Return only the answer the interviewee could say aloud."""

SYSTEM_PROMPT_ES = """Eres un desarrollador senior ayudando durante una entrevista técnica en vivo.
Responde en español natural, hablado, directo, técnicamente preciso, breve y fácil de decir en voz alta.
No empieces con relleno como 'Buena pregunta', 'Claro' o 'Por supuesto'.
Para preguntas conceptuales usa normalmente 2-4 frases cortas.
Para arquitectura o system design da el enfoque principal, el trade-off clave y un detalle práctico.
Para preguntas de código explica primero el enfoque y mantén el código muy corto salvo que el entrevistador pida una implementación completa.
Usa el contexto reciente para resolver preguntas de seguimiento y pronombres.
No inventes experiencia personal, empleadores, incidentes, métricas ni proyectos.
Si la transcripción contiene un pequeño error probable en un término técnico, infiere el término de software más plausible por contexto sin hablar del error de transcripción.
Devuelve únicamente la respuesta que el entrevistado podría decir en voz alta."""

ROUTER_PROMPT = """Decide whether the latest interviewer turn requires a technical software-engineering answer.
Return exactly ANSWER or IGNORE, with no punctuation or explanation.

ANSWER for questions or requests about programming, Java, Spring, APIs, microservices, databases, SQL, NoSQL, Kafka, cloud, DevOps, CI/CD, testing, security, concurrency, data structures, algorithms, debugging, architecture, system design, performance, coding, or concrete technical experience.

IGNORE for greetings, thanks, scheduling, salary, availability, introductions, generic small talk, or purely behavioral/non-technical prompts such as 'tell me about yourself'.

Use the recent interview context when the latest turn is a follow-up like 'why?', 'what about failures?', or 'and how would you scale it?'."""


@dataclass(frozen=True)
class ConversationTurn:
    speaker: str
    text: str


class GroqInterviewAssistant:
    """Groq GPT-OSS wrapper with routing and streaming answers."""

    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY no está configurada en el archivo .env."
            )

        self.client = Groq(api_key=api_key)
        self.model = GROQ_MODEL

    @staticmethod
    def _context_text(
        recent_turns: Iterable[ConversationTurn],
    ) -> str:
        lines: list[str] = []
        for turn in recent_turns:
            label = (
                "Interviewer"
                if turn.speaker == "INTERVIEWER"
                else "You"
            )
            lines.append(f"{label}: {turn.text}")
        return "\n".join(lines[-12:])

    def should_answer(
        self,
        question: str,
        recent_turns: Iterable[ConversationTurn],
    ) -> bool:
        context = self._context_text(recent_turns)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Recent interview conversation:\n{context}\n\n"
                        f"Latest interviewer turn:\n{question}"
                    ),
                },
            ],
            reasoning_effort="low",
            include_reasoning=False,
            max_completion_tokens=8,
            temperature=0.0,
        )

        text = (
            completion.choices[0].message.content or ""
        ).strip().upper()
        return text.startswith("ANSWER")

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
        context = self._context_text(recent_turns)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": instructions},
                {
                    "role": "user",
                    "content": (
                        "Recent interview conversation:\n"
                        f"{context}\n\n"
                        "Latest interviewer question:\n"
                        f"{question}\n\n"
                        "Answer the latest question. Use the recent "
                        "conversation only as context."
                    ),
                },
            ],
            reasoning_effort=GROQ_REASONING_EFFORT,
            include_reasoning=False,
            max_completion_tokens=GROQ_MAX_COMPLETION_TOKENS,
            temperature=0.25,
            stream=True,
        )

        collected: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue

            collected.append(delta)
            on_delta(delta)

        return "".join(collected).strip()
