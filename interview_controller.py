"""Stage 2 interview orchestration.

Reuses local Faster-Whisper audio capture while keeping completely independent
state from the Subtitle tab. Groq is called only after an interviewer turn is
considered complete.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable

from config import (
    INTERVIEW_CONTEXT_TURNS,
    INTERVIEW_QUESTION_DEBOUNCE_SECONDS,
    INTERVIEW_TRANSCRIPTION_LANGUAGE,
    INTERVIEW_WHISPER_INITIAL_PROMPT,
    INTERVIEW_WHISPER_MODEL,
    MIC_DEVICE_INDEX,
)
from interview_assistant import (
    ConversationTurn,
    GroqInterviewAssistant,
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
        self.assistant: GroqInterviewAssistant | None = None
        self.running = False

        self._turns: deque[ConversationTurn] = deque(
            maxlen=max(4, INTERVIEW_CONTEXT_TURNS)
        )
        self._pending_interviewer: list[str] = []
        self._pending_lock = threading.Lock()
        self._question_timer: threading.Timer | None = None

        self._answer_lock = threading.Lock()
        self._last_interviewer_text = ""
        self._last_turn_text = ""
        self._last_turn_speaker = ""
        self._last_question = ""
        self._last_answer = ""
        self._topic_memory = ""
        self._deferred_interviewer_text = ""

    def start(self) -> None:
        if self.running:
            return

        try:
            self.assistant = GroqInterviewAssistant()
        except Exception as exc:
            self._emit("on_status", f"Error: {exc}")
            return

        self.running = True
        self._turns.clear()
        self._pending_interviewer.clear()
        self._last_interviewer_text = ""
        self._last_turn_text = ""
        self._last_turn_speaker = ""
        self._last_question = ""
        self._last_answer = ""
        self._topic_memory = ""
        self._deferred_interviewer_text = ""

        callbacks = {
            "on_status": self._on_listener_status,
            "on_subtitle_partial": self._on_partial,
            "on_subtitle": self._on_final,
        }

        transcription_language = (
            None
            if INTERVIEW_TRANSCRIPTION_LANGUAGE == "auto"
            else INTERVIEW_TRANSCRIPTION_LANGUAGE
        )

        self.listener = ListenerController(
            callbacks,
            mic_device=self.mic_device,
            whisper_model=INTERVIEW_WHISPER_MODEL,
            transcription_language=transcription_language,
            initial_prompt=INTERVIEW_WHISPER_INITIAL_PROMPT,
        )
        self.listener.start()

    def stop(self) -> None:
        self.running = False

        with self._pending_lock:
            timer = self._question_timer
            self._question_timer = None
            self._pending_interviewer.clear()

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
        """Manually answer the latest complete turn.

        In solo testing, combine the latest consecutive YOU fragments because
        Faster-Whisper may split one spoken question into multiple finals.
        Automatic answers still only originate from INTERVIEWER turns.
        """
        question = self._last_interviewer_text.strip()
        if not question:
            question = self._recent_speaker_text("YOU", max_turns=4)

        if not question:
            self._emit(
                "on_assistant_error",
                "Todavía no hay una intervención para responder.",
            )
            return

        self._start_answer(question, force=True)

    def _recent_speaker_text(
        self,
        speaker: str,
        max_turns: int = 4,
    ) -> str:
        parts: list[str] = []
        for turn in reversed(self._turns):
            if turn.speaker != speaker:
                if parts:
                    break
                continue

            parts.append(turn.text)
            if len(parts) >= max_turns:
                break

        return " ".join(reversed(parts)).strip()

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

        # A long interviewer question may be split by the transcription
        # segmenter. Continuous partials mean the speaker is still talking, so
        # keep pushing the decision timer forward until speech actually stops.
        if speaker == "INTERVIEWER":
            self._extend_question_timer_if_pending()

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
        self._last_turn_text = cleaned
        self._last_turn_speaker = speaker

        self._turns.append(
            ConversationTurn(speaker=speaker, text=cleaned)
        )
        self._emit(
            "on_transcript_final",
            speaker,
            cleaned,
            segment_id,
        )

        if speaker == "INTERVIEWER":
            self._queue_interviewer_text(cleaned)

    def _new_question_timer(self) -> threading.Timer:
        timer = threading.Timer(
            INTERVIEW_QUESTION_DEBOUNCE_SECONDS,
            self._consume_interviewer_text,
        )
        timer.daemon = True
        return timer

    def _queue_interviewer_text(self, text: str) -> None:
        with self._pending_lock:
            self._pending_interviewer.append(text)

            if self._question_timer is not None:
                self._question_timer.cancel()

            self._question_timer = self._new_question_timer()
            self._question_timer.start()

    def _extend_question_timer_if_pending(self) -> None:
        with self._pending_lock:
            if not self._pending_interviewer:
                return

            if self._question_timer is not None:
                self._question_timer.cancel()

            self._question_timer = self._new_question_timer()
            self._question_timer.start()

    def _consume_interviewer_text(self) -> None:
        with self._pending_lock:
            parts = self._pending_interviewer[:]
            self._pending_interviewer.clear()
            self._question_timer = None

        if not self.running or not parts:
            return

        candidate = " ".join(parts).strip()
        if self._deferred_interviewer_text:
            candidate = (
                f"{self._deferred_interviewer_text} {candidate}"
            ).strip()
            self._deferred_interviewer_text = ""

        self._last_interviewer_text = candidate
        self._start_answer(candidate, force=False)

    def _start_answer(self, question: str, force: bool) -> None:
        if not self.running or self.assistant is None:
            return

        if not self._answer_lock.acquire(blocking=False):
            self._emit(
                "on_assistant_error",
                "El asistente todavía está generando la respuesta anterior.",
            )
            return

        recent_turns = list(self._turns)
        language = self.language

        self._emit("on_question_candidate", question)

        thread = threading.Thread(
            target=self._answer_worker,
            args=(question, recent_turns, language, force),
            name="interview-answer",
            daemon=True,
        )
        thread.start()

    def _answer_worker(
        self,
        question: str,
        recent_turns: list[ConversationTurn],
        language: str,
        force: bool,
    ) -> None:
        try:
            assistant = self.assistant
            if assistant is None or not self.running:
                return

            analysis = assistant.analyze_turn(
                raw_turn=question,
                recent_turns=recent_turns,
                topic_memory=self._topic_memory,
            )

            if analysis.topic or analysis.terms:
                memory_parts = []
                if analysis.topic:
                    memory_parts.append(f"Topic: {analysis.topic}")
                if analysis.terms:
                    memory_parts.append(
                        "Terms: " + ", ".join(analysis.terms)
                    )
                self._topic_memory = ". ".join(memory_parts)

            reconstructed = (
                analysis.question.strip()
                if analysis.question.strip()
                else question.strip()
            )

            self._emit(
                "on_turn_understood",
                reconstructed,
                analysis.topic,
                analysis.language,
            )

            if not force:
                if analysis.action == "WAIT":
                    self._deferred_interviewer_text = question.strip()
                    self._emit("on_question_waiting", reconstructed)
                    return

                if analysis.action != "ANSWER":
                    self._emit("on_question_ignored", reconstructed)
                    return

            self._last_question = reconstructed
            self._emit("on_question_detected", reconstructed)
            self._emit("on_answer_started", reconstructed)

            chunks: list[str] = []

            def on_delta(delta: str) -> None:
                chunks.append(delta)
                self._emit("on_answer_delta", delta)

            answer = assistant.stream_answer(
                question=reconstructed,
                recent_turns=recent_turns,
                language=language,
                on_delta=on_delta,
                topic_memory=self._topic_memory,
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
