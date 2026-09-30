"""Context-aware Groq engine for Stage 2 technical interviews.

Groq Whisper provides the raw transcript. GPT-OSS then interprets the latest
turn in the context of the ongoing interview, including active coding /
whiteboarding state, before deciding whether to answer.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Iterable

from groq import Groq

from config import (
    GROQ_CODING_MAX_COMPLETION_TOKENS,
    GROQ_MAX_COMPLETION_TOKENS,
    GROQ_MODEL,
    GROQ_REASONING_EFFORT,
    INTERVIEW_CODE_LANGUAGE,
)

DeltaCallback = Callable[[str], None]


SYSTEM_PROMPT_EN = """You are a senior software engineer helping during a live technical interview.
OUTPUT LANGUAGE IS LOCKED TO ENGLISH.
Always answer in English, regardless of whether the question, transcript, recent conversation, or topic memory is in Spanish or mixed language.
Do not translate the answer into Spanish unless the application explicitly selects Spanish before generating this response.
Write exactly like a strong human candidate would speak out loud.
Be direct, technically precise, concise, and conversational.
Default to ONE short paragraph of roughly 35-70 words.
Use at most 3 short sentences unless the interviewer explicitly asks for detail.
Do not use Markdown, headings, bullet lists, bold text, or filler such as 'Great question', 'Sure', or 'Absolutely'.
For simple definition/difference questions: definition + key distinction + one practical point.
For architecture/system-design questions: approach + main trade-off + one practical detail.
Use the ongoing interview context to resolve follow-ups and pronouns.
Do not invent personal experience, employers, incidents, metrics, or projects.
Return only the answer the interviewee could naturally say aloud."""

SYSTEM_PROMPT_ES = """Eres un desarrollador senior ayudando durante una entrevista técnica en vivo.
EL IDIOMA DE SALIDA ESTÁ BLOQUEADO EN ESPAÑOL.
Responde siempre en español, aunque la pregunta, transcripción, conversación reciente o memoria del tema estén en inglés o mezclen idiomas.
No respondas en inglés salvo que la aplicación seleccione explícitamente inglés antes de generar esta respuesta.
Escribe exactamente como respondería oralmente un candidato senior.
Sé directo, técnicamente preciso, breve y natural.
Por defecto responde en UN solo párrafo corto de unas 35-70 palabras.
Usa como máximo 3 frases cortas salvo que el entrevistador pida más detalle.
No uses Markdown, títulos, listas, negritas ni relleno como 'Buena pregunta', 'Claro' o 'Por supuesto'.
Para definiciones o diferencias: definición + diferencia clave + un punto práctico.
Para arquitectura/system design: enfoque + trade-off principal + un detalle práctico.
Usa el contexto continuo de la entrevista para resolver follow-ups y pronombres.
No inventes experiencia personal, empleadores, incidentes, métricas ni proyectos.
Devuelve únicamente la respuesta que el entrevistado podría decir de forma natural."""

CODING_PROMPT_EN = """You are a senior software engineer assisting during a live coding / whiteboarding interview.
OUTPUT LANGUAGE IS LOCKED TO ENGLISH.
The spoken explanation must be in English. Code identifiers and comments should use normal professional conventions.
Use the ACTIVE CODING CONTEXT. Follow-ups such as "implement it", "same with streams", "without extra memory", "optimize it", "what is the complexity?", and "what about nulls?" refer to the active problem unless a new problem is explicitly introduced.
Never silently switch to a different problem.
Preserve the previous solution's intent unless the interviewer asks for a different approach or constraint.
For an approach-only question, give a concise spoken approach and complexity; do not dump code unless requested.
When implementation or a modified implementation is requested, format the response exactly as plain text sections:
Approach:
<1-3 concise spoken sentences>
Complexity:
<time and space complexity>
Code:
<complete implementation, no Markdown code fences>
For complexity-only, testing-only, or edge-case follow-ups, answer only what was asked unless code is required.
Prefer simple interview-quality code over framework-heavy or clever code.
Do not invent requirements that were not stated.
Return content the candidate can quickly read and use during the interview."""

CODING_PROMPT_ES = """Eres un desarrollador senior ayudando durante una entrevista de coding / whiteboarding en vivo.
EL IDIOMA DE SALIDA ESTÁ BLOQUEADO EN ESPAÑOL.
La explicación oral debe estar en español. Los identificadores y comentarios del código deben seguir convenciones profesionales normales.
Usa el CONTEXTO DE CODING ACTIVO. Follow-ups como "impleméntalo", "haz lo mismo con streams", "sin memoria extra", "optimízalo", "cuál es la complejidad" o "qué pasa con null" se refieren al problema activo salvo que se introduzca explícitamente un problema nuevo.
Nunca cambies silenciosamente a otro problema.
Conserva la intención de la solución anterior salvo que el entrevistador pida otro enfoque o una nueva restricción.
Si solo piden el enfoque, da una explicación breve y la complejidad; no muestres código salvo que lo pidan.
Cuando pidan implementación o modificar la implementación, usa exactamente estas secciones de texto plano:
Approach:
<1-3 frases breves para decir oralmente>
Complexity:
<complejidad temporal y espacial>
Code:
<implementación completa, sin fences Markdown>
Para follow-ups solo de complejidad, pruebas o edge cases, responde únicamente lo solicitado salvo que haga falta código.
Prefiere código simple y apropiado para entrevista en lugar de soluciones innecesariamente complejas.
No inventes requisitos que no fueron indicados.
Devuelve contenido que el candidato pueda leer y usar rápidamente durante la entrevista."""

CONTEXTUALIZER_PROMPT = """You are the context and turn-understanding layer of a live software-engineering interview assistant.

The speech transcript can contain recognition mistakes, especially technical terms, acronyms, English spoken with an accent, or Spanish/English code-switching.

Your job is to understand the latest raw turn USING the recent conversation, current topic memory, and ACTIVE CODING CONTEXT.

Return ONLY valid JSON with this exact shape:
{
  "action": "ANSWER" | "IGNORE" | "WAIT",
  "question": "best reconstructed question, or empty string",
  "topic": "short current technical topic, or empty string",
  "language": "en" | "es" | "mixed" | "unknown",
  "terms": ["important technical terms"],
  "interview_type": "technical" | "coding" | "behavioral" | "other",
  "coding": {
    "new_problem": true | false,
    "problem": "active coding problem, or empty string",
    "request": "approach" | "implementation" | "modify_solution" | "optimization" | "complexity" | "testing" | "edge_cases" | "explanation" | "none",
    "constraints": ["explicit coding constraints introduced by the interviewer"],
    "language": "java" | "python" | "javascript" | "sql" | "other" | "unknown"
  }
}

Rules:
- Preserve the speaker's intended meaning; do not invent a new question.
- Correct obvious ASR mistakes only when phonetics plus context make the technical term reasonably clear.
- Examples: "J W T" / "jay double u tee" -> JWT; "oh auth" -> OAuth; "spring butt" -> Spring Boot.
- Treat GET, POST, PUT, PATCH and DELETE as HTTP/REST methods when the surrounding topic is APIs, controllers, endpoints or Spring Boot web development.
- In Spanish, phrases like "métodos en Spring Boot como get, post, update/patch" usually refer to HTTP methods and REST endpoint mappings, not Java methods or Spring lifecycle callbacks.
- Normalize spoken "update" to PUT/PATCH only when the context clearly refers to REST operations.
- Use previous turns to resolve fragments such as "and why?", "what about failures?", "and the other one?", or pronouns.
- Use ACTIVE CODING CONTEXT to resolve coding follow-ups such as "implement it", "same solution with streams", "without extra memory", "optimize that", "what is its complexity?", and "what about nulls?".
- Mark interview_type="coding" for algorithms, data-structure exercises, coding/whiteboarding problems, implementation requests, or follow-ups that modify/analyze the active coding solution.
- A conceptual question about Java, Streams, Spring, REST, etc. is interview_type="technical" unless it is tied to an active coding problem.
- Set coding.new_problem=true only when the interviewer introduces a genuinely new coding exercise. Follow-ups on the current problem must use false.
- For a coding follow-up, coding.problem should contain the resolved active problem when it is clear from context.
- coding.constraints must represent the FULL SET of constraints currently active after applying the latest follow-up, e.g. "without extra memory", "must use streams", "input may be null". If a newer instruction replaces an older constraint, omit the obsolete constraint.
- Behavioral questions about stakeholders, coworkers, conflict, leadership, teamwork, strengths, weaknesses, or similar personal-work stories must be interview_type="behavioral" and action="IGNORE".
- Greetings, thanks, scheduling, salary, company descriptions, and generic small talk must be interview_type="other" and action="IGNORE".
- ANSWER technical and coding questions/requests that can now be answered.
- WAIT when the latest turn sounds incomplete and more speech is likely needed.
- A technical/coding question may be in English, Spanish, or mixed.
- The reconstructed question should remain in the language in which the question was most likely asked.
- If uncertain about a corrupted technical term, keep the raw wording rather than confidently inventing a replacement.
"""


@dataclass(frozen=True)
class ConversationTurn:
    speaker: str
    text: str


@dataclass(frozen=True)
class CodingContext:
    problem: str = ""
    request: str = "none"
    constraints: tuple[str, ...] = ()
    language: str = INTERVIEW_CODE_LANGUAGE
    last_solution: str = ""


@dataclass(frozen=True)
class TurnAnalysis:
    action: str
    question: str
    topic: str
    language: str
    terms: tuple[str, ...]
    interview_type: str = "other"
    coding_new_problem: bool = False
    coding_problem: str = ""
    coding_request: str = "none"
    coding_constraints: tuple[str, ...] = ()
    coding_language: str = "unknown"


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

    @staticmethod
    def _coding_context_text(
        coding_context: CodingContext | None,
        include_solution: bool = False,
    ) -> str:
        if coding_context is None or not coding_context.problem:
            return "(none)"

        constraints = (
            ", ".join(coding_context.constraints)
            if coding_context.constraints
            else "(none)"
        )
        lines = [
            f"Problem: {coding_context.problem}",
            f"Current request: {coding_context.request}",
            f"Constraints: {constraints}",
            f"Implementation language: {coding_context.language}",
        ]

        if include_solution and coding_context.last_solution:
            # Keep enough of the previous answer to support transformations
            # such as "same solution with streams" without sending unbounded
            # session history.
            lines.append(
                "Previous generated solution:\n"
                + coding_context.last_solution[-6000:]
            )

        return "\n".join(lines)

    def analyze_turn(
        self,
        raw_turn: str,
        recent_turns: Iterable[ConversationTurn],
        topic_memory: str = "",
        coding_context: CodingContext | None = None,
    ) -> TurnAnalysis:
        """Interpret an imperfect transcript using ongoing interview context."""
        context = self._context_text(recent_turns)
        coding = self._coding_context_text(coding_context)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CONTEXTUALIZER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Current topic memory:\n{topic_memory or '(none)'}\n\n"
                        f"ACTIVE CODING CONTEXT:\n{coding}\n\n"
                        f"Recent interview conversation:\n{context or '(none)'}\n\n"
                        f"Latest raw transcript:\n{raw_turn}"
                    ),
                },
            ],
            reasoning_effort="low",
            include_reasoning=False,
            max_completion_tokens=320,
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

        interview_type = str(
            data.get("interview_type", "other")
        ).strip().lower()
        if interview_type not in {
            "technical",
            "coding",
            "behavioral",
            "other",
        }:
            interview_type = "other"

        coding_data = data.get("coding", {})
        if not isinstance(coding_data, dict):
            coding_data = {}

        coding_new_problem = bool(
            coding_data.get("new_problem", False)
        )
        coding_problem = " ".join(
            str(coding_data.get("problem", "")).strip().split()
        )
        coding_request = str(
            coding_data.get("request", "none")
        ).strip().lower()
        allowed_requests = {
            "approach",
            "implementation",
            "modify_solution",
            "optimization",
            "complexity",
            "testing",
            "edge_cases",
            "explanation",
            "none",
        }
        if coding_request not in allowed_requests:
            coding_request = "none"

        raw_constraints = coding_data.get("constraints", [])
        if not isinstance(raw_constraints, list):
            raw_constraints = []
        coding_constraints = tuple(
            " ".join(str(item).strip().split())
            for item in raw_constraints[:10]
            if str(item).strip()
        )

        coding_language = str(
            coding_data.get("language", "unknown")
        ).strip().lower()
        if coding_language not in {
            "java",
            "python",
            "javascript",
            "sql",
            "other",
            "unknown",
        }:
            coding_language = "unknown"

        # Behavioral help is intentionally outside this app's scope.
        if interview_type == "behavioral":
            action = "IGNORE"

        if action == "ANSWER" and not question:
            question = raw_turn.strip()

        return TurnAnalysis(
            action=action,
            question=question,
            topic=topic,
            language=language,
            terms=terms,
            interview_type=interview_type,
            coding_new_problem=coding_new_problem,
            coding_problem=coding_problem,
            coding_request=coding_request,
            coding_constraints=coding_constraints,
            coding_language=coding_language,
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
        interview_type: str = "technical",
        coding_context: CodingContext | None = None,
    ) -> str:
        is_coding = interview_type == "coding"

        if is_coding:
            instructions = (
                CODING_PROMPT_ES
                if language == "es"
                else CODING_PROMPT_EN
            )
            max_tokens = GROQ_CODING_MAX_COMPLETION_TOKENS
        else:
            instructions = (
                SYSTEM_PROMPT_ES
                if language == "es"
                else SYSTEM_PROMPT_EN
            )
            max_tokens = GROQ_MAX_COMPLETION_TOKENS

        output_language = (
            "SPANISH"
            if language == "es"
            else "ENGLISH"
        )
        context = self._context_text(recent_turns)
        coding = self._coding_context_text(
            coding_context,
            include_solution=is_coding,
        )

        user_content = (
            f"Current interview topic:\n{topic_memory or '(unknown)'}\n\n"
            f"Recent interview conversation:\n{context or '(none)'}\n\n"
            f"Contextually reconstructed interviewer question:\n{question}\n\n"
        )

        if is_coding:
            user_content += (
                f"ACTIVE CODING CONTEXT:\n{coding}\n\n"
                "Use the active problem and previous generated solution to "
                "resolve this follow-up. If the interviewer asks for the same "
                "solution using a different technique (for example Java "
                "Streams), produce the updated complete implementation rather "
                "than explaining the technique in isolation.\n\n"
            )

        user_content += (
            f"MANDATORY OUTPUT LANGUAGE: {output_language}.\n"
            "The output-language setting overrides the language used by the "
            "interviewer and by the conversation context. Answer the request "
            "using the recent conversation to resolve follow-ups."
        )

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_content},
            ],
            reasoning_effort=GROQ_REASONING_EFFORT,
            include_reasoning=False,
            max_completion_tokens=max_tokens,
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
