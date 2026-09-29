"""Stage 2 interview orchestration.

Reuses the local audio/transcription pipeline but keeps independent state from
the Subtitle tab. Only remote/interviewer questions invoke OpenAI.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable

from config import (
    INTERVIEW_CONTEXT_TURNS,
    INTERVIEW_QUESTION_DEBOUNCE_SECONDS,
    MIC_DEVICE_INDEX,
)
from interview_assistant import (
    ConversationTurn,
    OpenAIInterviewAssistant,
    looks_like_question,
)
from vad_detector import ListenerController


class InterviewController:
    def __init__(
        self,
        callbacks: dict[str, Callable],
        mic_device: int = MIC_DEVICE_INDEX,
        language: str = "en",
    ) -> None:
        self.callbacks = callbacks
        self.mic_device = mic_device
        self.language = language if language in {"en", "es"} else "en"

        self.listener: ListenerController | None = None
        self.assistant: OpenAIInterviewAssistant | None = None
        self.running = False

        self._turns: deque[ConversationTurn] = deque(
            maxlen=max(4, INTERVIEW_CONTEXT_TURNS)
        )
        self._pending_interviewer: list[str] = []
        self._pending_lock = threading.Lock()
        self._question_timer: threading.Timer | None = None

        self._answer_lock = threading.Lock()
        self._last_interviewer_text = ""
        self._last_question = ""
        self._last_answer = ""

    def start(self) -> None:
        if self.running:
            return

        try:
            self.assistant = OpenAIInterviewAssistant()
        except Exception as exc:
            self._emit("on_status", f"Error: {exc}")
            return

        self.running = True
        self._turns.clear()
        self._pending_interviewer.clear()
        self._last_interviewer_text = ""
        self._last_question = ""
        self._last_answer = ""

        callbacks = {
            "on_status": self._on_listener_status,
            "on_subtitle_partial": self._on_partial,
            "on_subtitle": self._on_final,
        }

        self.listener = ListenerController(
            callbacks,
            mic_device=self.mic_device,
        )
        self.listener.start()

    def stop(self) -> None:
        self.running = False

        timer = self._question_timer
        self._question_timer = None
        if timer is not None:
            timer.cancel()

        listener = self.listener
        self.listener = None
        if listener is not None:
            listener.stop()

        self.assistant = None
        self._emit("on_status", "Detenido")

    def set_language(self, language: str) -> None:
        if language in {"en", "es"}:
            self.language = language
            self._emit("on_language_changed", language)

    def answer_last_interviewer_turn(self) -> None:
        question = self._last_interviewer_text.strip()
        if not question:
            self._emit(
                "on_assistant_error",
                "Todavía no hay una intervención del entrevistador.",
            )
            return

        self._start_answer(question, force=True)

    def regenerate(self) -> None:
        question = self._last_question.strip()
        if not question:
            self.answer_last_interviewer_turn()
            return
        self._start_answer(question, force=True)

    def _on_listener_status(self, text: str) -> None:
        self._emit("on_status", text)

    def _on_partial(
        self,
        source: str,
        text: str,
        segment_id: int,
    ) -> None:
        speaker = "YOU" if source == "YOU" else "INTERVIEWER"
        self._emit(
            "on_transcript_partial",
            speaker,
            text,
            segment_id,
        )

    def _on_final(
        self,
        source: str,
        text: str,
        segment_id: int,
    ) -> None:
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            return

        speaker = "YOU" if source == "YOU" else "INTERVIEWER"
        self._turns.append(ConversationTurn(speaker=speaker, text=cleaned))
        self._emit(
            "on_transcript_final",
            speaker,
            cleaned,
            segment_id,
        )

        if speaker == "INTERVIEWER":
            self._last_interviewer_text = cleaned
            self._queue_interviewer_text(cleaned)

    def _queue_interviewer_text(self, text: str) -> None:
        with self._pending_lock:
            self._pending_interviewer.append(text)

            if self._question_timer is not None:
                self._question_timer.cancel()

            self._question_timer = threading.Timer(
                INTERVIEW_QUESTION_DEBOUNCE_SECONDS,
                self._consume_interviewer_text,
            )
            self._question_timer.daemon = True
            self._question_timer.start()

    def _consume_interviewer_text(self) -> None:
        with self._pending_lock:
            parts = self._pending_interviewer[:]
            self._pending_interviewer.clear()
            self._question_timer = None

        if not self.running or not parts:
            return

        candidate = " ".join(parts).strip()
        self._last_interviewer_text = candidate

        if looks_like_question(candidate):
            self._start_answer(candidate, force=False)

    def _start_answer(self, question: str, force: bool) -> None:
        if not self.running or self.assistant is None:
            return

        if not force and not looks_like_question(question):
            return

        if not self._answer_lock.acquire(blocking=False):
            # Avoid overlapping model responses during a fast back-and-forth.
            self._emit(
                "on_assistant_error",
                "El asistente todavía está generando la respuesta anterior.",
            )
            return

        recent_turns = list(self._turns)
        language = self.language
        self._last_question = question

        self._emit("on_question_detected", question)
        self._emit("on_answer_started", question)

        thread = threading.Thread(
            target=self._answer_worker,
            args=(question, recent_turns, language),
            name="interview-answer",
            daemon=True,
        )
        thread.start()

    def _answer_worker(
        self,
        question: str,
        recent_turns: list[ConversationTurn],
        language: str,
    ) -> None:
        try:
            chunks: list[str] = []

            def on_delta(delta: str) -> None:
                chunks.append(delta)
                self._emit("on_answer_delta", delta)

            assert self.assistant is not None
            answer = self.assistant.stream_answer(
                question=question,
                recent_turns=recent_turns,
                language=language,
                on_delta=on_delta,
            )

            self._last_answer = answer or "".join(chunks).strip()
            self._emit(
                "on_answer_completed",
                self._last_answer,
            )
        except Exception as exc:
            self._emit("on_assistant_error", str(exc))
        finally:
            self._answer_lock.release()

    def _emit(self, name: str, *args) -> None:
        callback = self.callbacks.get(name)
        if callback:
            callback(*args)
