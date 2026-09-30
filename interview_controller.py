"""Stage 2 interview orchestration.

Reuses local Faster-Whisper audio capture while keeping completely independent
state from the Subtitle tab. Groq is called only after an interviewer turn is
considered complete.
"""

from __future__ import annotations

import re
import threading
from collections import Counter, deque
from typing import Callable

from config import (
    ASSISTANT_RESPONSE_SCOPE,
    GROQ_STT_MODEL,
    INTERVIEW_CONTEXT_TURNS,
    INTERVIEW_MAX_UTTERANCE_SECONDS,
    INTERVIEW_QUESTION_DEBOUNCE_SECONDS,
    INTERVIEW_SPEECH_END_MS,
    INTERVIEW_STT_PROVIDER,
    INTERVIEW_TRANSCRIPTION_LANGUAGE,
    INTERVIEW_WHISPER_INITIAL_PROMPT,
    INTERVIEW_WHISPER_MODEL,
    MIC_DEVICE_INDEX,
)
from interview_assistant import (
    ConversationTurn,
    GroqInterviewAssistant,
)
from groq_health import describe_groq_error
from vad_detector import ListenerController


class InterviewController:
    def __init__(
        self,
        callbacks: dict[str, Callable],
        mic_device: int = MIC_DEVICE_INDEX,
        language: str = "en",
        response_scope: str = ASSISTANT_RESPONSE_SCOPE,
    ) -> None:
        self.callbacks = callbacks
        self.mic_device = mic_device
        self.language = language if language in {"en", "es"} else "en"
        self.response_scope = (
            response_scope
            if response_scope in {"interviewer", "both"}
            else "interviewer"
        )

        self.listener: ListenerController | None = None
        self.assistant: GroqInterviewAssistant | None = None
        self.running = False

        self._turns: deque[ConversationTurn] = deque(
            maxlen=max(4, INTERVIEW_CONTEXT_TURNS)
        )
        self._pending_response_parts: list[str] = []
        self._pending_response_speaker = ""
        self._pending_lock = threading.Lock()
        self._question_timer: threading.Timer | None = None

        self._answer_lock = threading.Lock()
        self._last_interviewer_text = ""
        self._last_turn_text = ""
        self._last_turn_speaker = ""
        self._last_question = ""
        self._last_answer = ""
        self._topic_memory = ""
        self._deferred_response_text = ""
        self._deferred_response_speaker = ""

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
        self._pending_response_parts.clear()
        self._pending_response_speaker = ""
        self._last_interviewer_text = ""
        self._last_turn_text = ""
        self._last_turn_speaker = ""
        self._last_question = ""
        self._last_answer = ""
        self._topic_memory = ""
        self._deferred_response_text = ""
        self._deferred_response_speaker = ""

        callbacks = {
            "on_status": self._on_listener_status,
            "on_subtitle_partial": self._on_partial,
            "on_subtitle": self._on_final,
            "on_transcription_error": self._on_transcription_error,
        }

        transcription_language = (
            None
            if INTERVIEW_TRANSCRIPTION_LANGUAGE == "auto"
            else INTERVIEW_TRANSCRIPTION_LANGUAGE
        )

        transcriber_factory = None
        partials_enabled = True
        engine_label = "Faster-Whisper local"

        if INTERVIEW_STT_PROVIDER == "groq":
            from groq_transcriber import GroqSpeechTranscriber

            def transcriber_factory(**kwargs):
                return GroqSpeechTranscriber(
                    language=transcription_language,
                    **kwargs,
                )

            # For Stage 2 we prefer a clean final question after a natural
            # pause instead of noisy rolling partial text.
            partials_enabled = False
            engine_label = f"Groq {GROQ_STT_MODEL}"

        self.listener = ListenerController(
            callbacks,
            mic_device=self.mic_device,
            whisper_model=INTERVIEW_WHISPER_MODEL,
            transcription_language=transcription_language,
            initial_prompt=INTERVIEW_WHISPER_INITIAL_PROMPT,
            transcriber_factory=transcriber_factory,
            speech_end_ms=INTERVIEW_SPEECH_END_MS,
            max_utterance_seconds=INTERVIEW_MAX_UTTERANCE_SECONDS,
            partials_enabled=partials_enabled,
            engine_label=engine_label,
        )
        self.listener.start()

    def stop(self) -> None:
        self.running = False

        with self._pending_lock:
            timer = self._question_timer
            self._question_timer = None
            self._pending_response_parts.clear()
            self._pending_response_speaker = ""

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

    def set_response_scope(self, scope: str) -> None:
        if scope not in {"interviewer", "both"}:
            return

        self.response_scope = scope

        # If the user switches back to interviewer-only mode while a local
        # YOU turn is waiting for analysis, discard that pending trigger. The
        # transcript itself remains in context.
        if scope == "interviewer":
            with self._pending_lock:
                if self._pending_response_speaker == "YOU":
                    if self._question_timer is not None:
                        self._question_timer.cancel()
                    self._question_timer = None
                    self._pending_response_parts.clear()
                    self._pending_response_speaker = ""

                if self._deferred_response_speaker == "YOU":
                    self._deferred_response_text = ""
                    self._deferred_response_speaker = ""

        self._emit("on_response_scope_changed", scope)

    def answer_last_interviewer_turn(self) -> None:
        """Manually answer the latest turn allowed by the response scope."""
        if self.response_scope == "interviewer":
            question = self._last_interviewer_text.strip()
            if not question:
                self._emit(
                    "on_assistant_error",
                    "Todavía no hay una intervención del entrevistador para responder.",
                )
                return
        else:
            question = self._last_turn_text.strip()
            if not question:
                self._emit(
                    "on_assistant_error",
                    "Todavía no hay una intervención para responder.",
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

    def _on_transcription_error(
        self,
        source: str,
        exc: Exception,
    ) -> None:
        if INTERVIEW_STT_PROVIDER == "groq":
            message = describe_groq_error(
                exc,
                f"Whisper {GROQ_STT_MODEL}",
            )
        else:
            message = f"Transcription {source}: {exc}"

        self._emit("on_service_error", message)

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

        # Keep the decision timer open while an allowed speaker is still
        # talking. In interviewer mode, YOU never triggers an answer.
        if self._speaker_can_trigger(speaker):
            self._extend_question_timer_if_pending(speaker)

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

        if self._is_low_quality_transcript(cleaned):
            print(
                f"[Stage2:{speaker}] Descarta transcripción ruidosa: "
                f"{cleaned[:160]}"
            )
            self._emit(
                "on_transcript_rejected",
                speaker,
                cleaned,
            )
            return

        self._last_turn_text = cleaned
        self._last_turn_speaker = speaker
        if speaker == "INTERVIEWER":
            self._last_interviewer_text = cleaned

        self._turns.append(
            ConversationTurn(speaker=speaker, text=cleaned)
        )
        self._emit(
            "on_transcript_final",
            speaker,
            cleaned,
            segment_id,
        )

        if self._speaker_can_trigger(speaker):
            self._queue_response_text(speaker, cleaned)

    @staticmethod
    def _is_low_quality_transcript(text: str) -> bool:
        """Reject obvious ASR loops before they pollute interview context."""
        tokens = [
            re.sub(r"[^a-z0-9áéíóúüñ]+", "", token.casefold())
            for token in text.split()
        ]
        tokens = [token for token in tokens if token]

        if len(tokens) < 8:
            return False

        counts = Counter(tokens)
        dominant_ratio = counts.most_common(1)[0][1] / len(tokens)

        longest_run = 1
        current_run = 1
        for previous, current in zip(tokens, tokens[1:]):
            if current == previous:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 1

        unique_ratio = len(counts) / len(tokens)

        return (
            dominant_ratio >= 0.45
            or longest_run >= 4
            or (len(tokens) >= 14 and unique_ratio <= 0.25)
        )

    def _speaker_can_trigger(self, speaker: str) -> bool:
        return (
            speaker == "INTERVIEWER"
            or self.response_scope == "both"
        )

    def _new_question_timer(self) -> threading.Timer:
        timer = threading.Timer(
            INTERVIEW_QUESTION_DEBOUNCE_SECONDS,
            self._consume_response_text,
        )
        timer.daemon = True
        return timer

    def _queue_response_text(
        self,
        speaker: str,
        text: str,
    ) -> None:
        flush_speaker = ""
        flush_candidate = ""

        with self._pending_lock:
            # Never merge speech from two different people into one question.
            if (
                self._pending_response_parts
                and self._pending_response_speaker
                and self._pending_response_speaker != speaker
            ):
                flush_speaker = self._pending_response_speaker
                flush_candidate = " ".join(
                    self._pending_response_parts
                ).strip()
                self._pending_response_parts.clear()

            self._pending_response_speaker = speaker
            self._pending_response_parts.append(text)

            if self._question_timer is not None:
                self._question_timer.cancel()

            self._question_timer = self._new_question_timer()
            self._question_timer.start()

        if flush_candidate:
            self._dispatch_candidate(
                flush_speaker,
                flush_candidate,
            )

    def _extend_question_timer_if_pending(
        self,
        speaker: str,
    ) -> None:
        with self._pending_lock:
            if not self._pending_response_parts:
                return
            if self._pending_response_speaker != speaker:
                return

            if self._question_timer is not None:
                self._question_timer.cancel()

            self._question_timer = self._new_question_timer()
            self._question_timer.start()

    def _consume_response_text(self) -> None:
        with self._pending_lock:
            parts = self._pending_response_parts[:]
            speaker = self._pending_response_speaker
            self._pending_response_parts.clear()
            self._pending_response_speaker = ""
            self._question_timer = None

        if not self.running or not parts or not speaker:
            return

        candidate = " ".join(parts).strip()
        self._dispatch_candidate(speaker, candidate)

    def _dispatch_candidate(
        self,
        speaker: str,
        candidate: str,
    ) -> None:
        if not self.running or not candidate:
            return
        if not self._speaker_can_trigger(speaker):
            return

        if (
            self._deferred_response_text
            and self._deferred_response_speaker == speaker
        ):
            candidate = (
                f"{self._deferred_response_text} {candidate}"
            ).strip()
            self._deferred_response_text = ""
            self._deferred_response_speaker = ""

        if speaker == "INTERVIEWER":
            self._last_interviewer_text = candidate

        self._start_answer(candidate, force=False, speaker=speaker)

    def _start_answer(
        self,
        question: str,
        force: bool,
        speaker: str = "",
    ) -> None:
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
            args=(question, recent_turns, language, force, speaker),
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
        speaker: str,
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
                    self._deferred_response_text = question.strip()
                    self._deferred_response_speaker = speaker
                    self._emit("on_question_waiting", reconstructed)
                    return

                if analysis.action != "ANSWER":
                    self._emit("on_question_ignored", reconstructed)
                    return

            self._last_question = reconstructed
            self._emit("on_question_detected", reconstructed)
            self._emit("on_answer_started", reconstructed)

            generated_answer = ""

            # A rare Groq stream can finish successfully without returning
            # usable content. Retry exactly once with the same question,
            # context and output language before giving up.
            for attempt in range(2):
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

                generated_answer = (
                    answer or "".join(chunks)
                ).strip()

                if generated_answer:
                    break

                if attempt == 0 and self.running:
                    print(
                        "[Stage2] Groq devolvió una respuesta vacía; "
                        "reintentando una vez."
                    )
                    self._emit("on_answer_retrying")

            if generated_answer:
                self._last_answer = generated_answer

            self._emit(
                "on_answer_completed",
                generated_answer,
            )
        except Exception as exc:
            self._emit(
                "on_assistant_error",
                describe_groq_error(
                    exc,
                    "GPT " + str(getattr(self.assistant, "model", "Groq")),
                ),
            )
        finally:
            self._answer_lock.release()

    def _emit(self, name: str, *args) -> None:
        callback = self.callbacks.get(name)
        if callback:
            callback(*args)
