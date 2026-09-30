"""Context-aware Groq engine for Stage 2 technical interviews.

Faster-Whisper provides a fast raw transcript. Groq then interprets that raw
turn in the context of the ongoing interview before deciding whether to answer.
"""

from __future__ import annotations

import json
import os
import re
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
Use the ongoing interview context to resolve follow-up questions and pronouns.
Do not invent personal experience, employers, incidents, metrics, or projects.
Return only the answer the interviewee could say aloud."""

SYSTEM_PROMPT_ES = """Eres un desarrollador senior ayudando durante una entrevista técnica en vivo.
Responde en español natural, hablado, directo, técnicamente preciso, breve y fácil de decir en voz alta.
No empieces con relleno como 'Buena pregunta', 'Claro' o 'Por supuesto'.
Para preguntas conceptuales usa normalmente 2-4 frases cortas.
Para arquitectura o system design da el enfoque principal, el trade-off clave y un detalle práctico.
Para preguntas de código explica primero el enfoque y mantén el código muy corto salvo que el entrevistador pida una implementación completa.
Usa el contexto continuo de la entrevista para resolver preguntas de seguimiento y pronombres.
No inventes experiencia personal, empleadores, incidentes, métricas ni proyectos.
Devuelve únicamente la respuesta que el entrevistado podría decir en voz alta."""

CONTEXTUALIZER_PROMPT = """You are the context and turn-understanding layer of a live software-engineering interview assistant.

The speech transcript can contain recognition mistakes, especially technical terms, acronyms, English spoken with an accent, or Spanish/English code-switching.

Your job is to understand the latest raw turn USING the recent conversation and the current topic memory.

Return ONLY valid JSON with this exact shape:
{
  "action": "ANSWER" | "IGNORE" | "WAIT",
  "question": "best reconstructed technical question, or empty string",
  "topic": "short current technical topic, or empty string",
  "language": "en" | "es" | "mixed" | "unknown",
  "terms": ["important technical terms"]
}

Rules:
- Preserve the speaker's intended meaning; do not invent a new question.
- Correct obvious ASR mistakes only when phonetics plus context make the technical term reasonably clear.
- Examples: "J W T" / "jay double u tee" -> JWT; "oh auth" -> OAuth; "spring butt" -> Spring Boot.
- Use previous turns to resolve fragments such as "and why?", "what about failures?", "and the other one?", or pronouns.
- ANSWER when the turn is a technical software-engineering question/request that can now be answered.
- WAIT when the latest turn sounds incomplete and more speech is likely needed.
- IGNORE for greetings, thanks, scheduling, salary, generic small talk, or non-technical conversation.
- A technical question may be in English, Spanish, or mixed.
- The reconstructed question should remain in the language in which the question was most likely asked.
- If uncertain about a corrupted technical term, keep the raw wording rather than confidently inventing a replacement.
"""


@dataclass(frozen=True)
class ConversationTurn:
    speaker: str
    text: str


@dataclass(frozen=True)
class TurnAnalysis:
    action: str
    question: str
    topic: str
    language: str
    terms: tuple[str, ...]


class GroqInterviewAssistant:
    """Groq GPT-OSS wrapper with contextual turn analysis and streaming answers."""

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
        return "\n".join(lines[-24:])

    def analyze_turn(
        self,
        raw_turn: str,
        recent_turns: Iterable[ConversationTurn],
        topic_memory: str = "",
    ) -> TurnAnalysis:
        """Interpret an imperfect transcript using the ongoing interview context."""
        context = self._context_text(recent_turns)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CONTEXTUALIZER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Current topic memory:\n{topic_memory or '(none)'}\n\n"
                        f"Recent interview conversation:\n{context or '(none)'}\n\n"
                        f"Latest raw transcript:\n{raw_turn}"
                    ),
                },
            ],
            reasoning_effort="low",
            include_reasoning=False,
            max_completion_tokens=220,
            temperature=0.0,
        )

        raw = (completion.choices[0].message.content or "").strip()
        data = self._parse_analysis_json(raw)

        action = str(data.get("action", "IGNORE")).upper()
        if action not in {"ANSWER", "IGNORE", "WAIT"}:
            action = "IGNORE"

        question = " ".join(
            str(data.get("question", "")).strip().split()
        )
        topic = " ".join(
            str(data.get("topic", "")).strip().split()
        )
        language = str(data.get("language", "unknown")).lower()
        if language not in {"en", "es", "mixed", "unknown"}:
            language = "unknown"

        raw_terms = data.get("terms", [])
        if not isinstance(raw_terms, list):
            raw_terms = []
        terms = tuple(
            str(term).strip()
            for term in raw_terms[:12]
            if str(term).strip()
        )

        if action == "ANSWER" and not question:
            question = raw_turn.strip()

        return TurnAnalysis(
            action=action,
            question=question,
            topic=topic,
            language=language,
            terms=terms,
        )

    @staticmethod
    def _parse_analysis_json(text: str) -> dict:
        cleaned = text.strip()
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                return {}
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}

    def stream_answer(
        self,
        question: str,
        recent_turns: Iterable[ConversationTurn],
        language: str,
        on_delta: DeltaCallback,
        topic_memory: str = "",
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
                        f"Current interview topic:\n{topic_memory or '(unknown)'}\n\n"
                        "Recent interview conversation:\n"
                        f"{context or '(none)'}\n\n"
                        "Contextually reconstructed interviewer question:\n"
                        f"{question}\n\n"
                        "Answer that question. Use the recent conversation "
                        "to resolve follow-ups, but do not repeat the transcript."
                    ),
                },
            ],
            reasoning_effort=GROQ_REASONING_EFFORT,
            include_reasoning=False,
            max_completion_tokens=GROQ_MAX_COMPLETION_TOKENS,
            temperature=0.2,
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
