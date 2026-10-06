"""Stage 2 interview orchestration.

Reuses local Faster-Whisper audio capture while keeping completely independent
state from the Subtitle tab. Groq is called only after an interviewer turn is
considered complete.
"""

from __future__ import annotations

import re
import threading
import time
from collections import Counter, deque
from typing import Callable

from config import (
    ASSISTANT_RESPONSE_SCOPE,
    GROQ_STT_MODEL,
    INTERVIEW_CODE_LANGUAGE,
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
    CodingContext,
    ConversationTurn,
    GroqInterviewAssistant,
)
from groq_health import describe_groq_error
from question_assembler import QuestionAssembler, QuestionCandidate
from vad_detector import ListenerController


class InterviewController:
    def __init__(
        self,
        callbacks: dict[str, Callable],
        mic_device: int = MIC_DEVICE_INDEX,
        language: str = "en",
        response_scope: str = ASSISTANT_RESPONSE_SCOPE,
        transcription_language: str = INTERVIEW_TRANSCRIPTION_LANGUAGE,
    ) -> None:
        self.callbacks = callbacks
        self.mic_device = mic_device
        self.language = language if language in {"en", "es"} else "en"
        self.response_scope = (
            response_scope
            if response_scope in {"interviewer", "both"}
            else "interviewer"
        )
        self.transcription_language = (
            transcription_language
            if transcription_language in {"auto", "en", "es"}
            else "auto"
        )

        self.listener: ListenerController | None = None
        self.assistant: GroqInterviewAssistant | None = None
        self.running = False

        self._turns: deque[ConversationTurn] = deque(
            maxlen=max(4, INTERVIEW_CONTEXT_TURNS)
        )
        self._question_assembler = QuestionAssembler()
        self._pending_lock = threading.Lock()
        self._question_timer: threading.Timer | None = None

        self._answer_lock = threading.Lock()
        self._queued_answer_text = ""
        self._queued_answer_speaker = ""
        self._queued_answer_force = False
        self._queued_answer_candidate: QuestionCandidate | None = None
        self._last_interviewer_text = ""
        self._last_turn_text = ""
        self._last_turn_speaker = ""
        self._last_question = ""
        self._last_answer = ""
        self._coding_context = CodingContext(
            language=INTERVIEW_CODE_LANGUAGE
        )
        self._topic_memory = ""

    @staticmethod
    def _diag(event: str, **fields) -> None:
        """Metadata-only Stage 2 diagnostics; never prints transcript text."""
        stamp = time.strftime("%H:%M:%S")
        details = " ".join(
            f"{key}={value}"
            for key, value in fields.items()
            if value not in {"", None}
        )
        suffix = f" {details}" if details else ""
        print(f"[Stage2Diag {stamp}] {event}{suffix}")

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
        self._question_assembler.clear()
        self._last_interviewer_text = ""
        self._last_turn_text = ""
        self._last_turn_speaker = ""
        self._last_question = ""
        self._last_answer = ""
        self._queued_answer_text = ""
        self._queued_answer_speaker = ""
        self._queued_answer_force = False
        self._queued_answer_candidate = None
        self._coding_context = CodingContext(
            language=INTERVIEW_CODE_LANGUAGE
        )
        self._topic_memory = ""

        callbacks = {
            "on_status": self._on_listener_status,
            "on_subtitle_partial": self._on_partial,
            "on_subtitle": self._on_final,
            "on_transcription_error": self._on_transcription_error,
        }

        transcription_language = (
            None
            if self.transcription_language == "auto"
            else self.transcription_language
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
            self._question_assembler.clear()
            self._queued_answer_text = ""
            self._queued_answer_speaker = ""
            self._queued_answer_force = False
            self._queued_answer_candidate = None

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

    def set_transcription_language(self, language: str) -> None:
        if language not in {"auto", "en", "es"}:
            return

        self.transcription_language = language
        resolved_language = None if language == "auto" else language

        listener = self.listener
        if listener is not None:
            listener.transcription_language = resolved_language
            transcriber = listener.transcriber
            if transcriber is not None and hasattr(transcriber, "language"):
                transcriber.language = resolved_language

    def set_response_scope(self, scope: str) -> None:
        if scope not in {"interviewer", "both"}:
            return

        self.response_scope = scope

        # If the user switches back to interviewer-only mode while a local
        # YOU turn is being assembled, discard that pending trigger. The
        # transcript itself remains in the conversation history.
        if scope == "interviewer":
            with self._pending_lock:
                candidate = self._question_assembler.snapshot()
                if candidate is not None and candidate.speaker == "YOU":
                    if self._question_timer is not None:
                        self._question_timer.cancel()
                    self._question_timer = None
                    self._question_assembler.clear()

                if (
                    self._queued_answer_candidate is not None
                    and self._queued_answer_candidate.speaker == "YOU"
                ):
                    self._queued_answer_text = ""
                    self._queued_answer_speaker = ""
                    self._queued_answer_force = False
                    self._queued_answer_candidate = None

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
        prefix = "__STAGE2_LATENCY__:"
        if text.startswith(prefix):
            try:
                _, stage, value = text.split(":", 2)
                latency_ms = float(value)
            except (ValueError, TypeError):
                return

            self._emit("on_latency", stage, latency_ms)
            self._diag(
                "latency",
                stage=stage,
                ms=f"{latency_ms:.1f}",
            )
            return

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
        self._diag(
            "stt_final",
            speaker=speaker,
            segment=segment_id,
            chars=len(cleaned),
        )

        if self._is_low_quality_transcript(cleaned):
            self._diag(
                "stt_rejected",
                speaker=speaker,
                segment=segment_id,
                chars=len(cleaned),
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
        """Reject obvious ASR hallucinations before they pollute context."""
        normalized = " ".join(
            re.sub(
                r"[^a-z0-9áéíóúüñ]+",
                " ",
                text.casefold(),
            ).split()
        )

        # Whisper commonly emits these short courtesy/video phrases when a
        # segment contains noise, silence, room audio, or indistinct speech.
        # They carry no useful technical-interview context, so reject them
        # before they reach the conversation history or contextualizer.
        noise_phrases = {
            "thank you",
            "thank you very much",
            "thank you so much",
            "thanks",
            "thanks a lot",
            "thanks so much",
            "thank you for watching",
            "thanks for watching",
            "thank you for listening",
            "thanks for listening",
        }
        if normalized in noise_phrases:
            return True

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
        boundary_candidate: QuestionCandidate | None = None
        current_candidate: QuestionCandidate | None = None

        with self._pending_lock:
            boundary_candidate, current_candidate = (
                self._question_assembler.append(
                    speaker,
                    text,
                )
            )

            if self._question_timer is not None:
                self._question_timer.cancel()

            self._question_timer = self._new_question_timer()
            self._question_timer.start()

        if boundary_candidate is not None:
            self._diag(
                "assembly_boundary",
                assembly=boundary_candidate.assembly_id,
                revision=boundary_candidate.revision,
                speaker=boundary_candidate.speaker,
                parts=boundary_candidate.part_count,
            )
            self._dispatch_candidate(boundary_candidate)

        if current_candidate is not None:
            event = (
                "assembly_started"
                if current_candidate.revision == 1
                else "assembly_extended"
            )
            self._diag(
                event,
                assembly=current_candidate.assembly_id,
                revision=current_candidate.revision,
                speaker=current_candidate.speaker,
                parts=current_candidate.part_count,
                chars=len(current_candidate.text),
            )

    def _extend_question_timer_if_pending(
        self,
        speaker: str,
    ) -> None:
        with self._pending_lock:
            if not self._question_assembler.has_pending(speaker):
                return

            if self._question_timer is not None:
                self._question_timer.cancel()

            self._question_timer = self._new_question_timer()
            self._question_timer.start()

    def _consume_response_text(self) -> None:
        with self._pending_lock:
            candidate = self._question_assembler.snapshot()
            self._question_timer = None

        if not self.running or candidate is None:
            return

        self._dispatch_candidate(candidate)

    def _dispatch_candidate(
        self,
        candidate: QuestionCandidate,
    ) -> None:
        if not self.running or not candidate.text:
            return
        if not self._speaker_can_trigger(candidate.speaker):
            return

        self._last_turn_text = candidate.text
        self._last_turn_speaker = candidate.speaker
        if candidate.speaker == "INTERVIEWER":
            self._last_interviewer_text = candidate.text

        self._start_answer(
            candidate.text,
            force=False,
            speaker=candidate.speaker,
            candidate=candidate,
        )

    def _candidate_is_stale(
        self,
        candidate: QuestionCandidate | None,
    ) -> bool:
        if candidate is None:
            return False

        with self._pending_lock:
            return self._question_assembler.is_stale(candidate)

    def _claim_candidate(
        self,
        candidate: QuestionCandidate | None,
    ) -> bool:
        if candidate is None:
            return True

        with self._pending_lock:
            active = self._question_assembler.snapshot()
            was_active = (
                active is not None
                and active.assembly_id == candidate.assembly_id
            )
            claimed = self._question_assembler.claim(candidate)

            if claimed and was_active:
                if self._question_timer is not None:
                    self._question_timer.cancel()
                self._question_timer = None

            return claimed

    def _discard_candidate(
        self,
        candidate: QuestionCandidate | None,
    ) -> None:
        if candidate is None:
            return

        with self._pending_lock:
            active = self._question_assembler.snapshot()
            was_active = (
                active is not None
                and active.assembly_id == candidate.assembly_id
                and active.revision == candidate.revision
            )
            self._question_assembler.discard(candidate)

            if was_active:
                if self._question_timer is not None:
                    self._question_timer.cancel()
                self._question_timer = None

    def _recent_turns_for_candidate(
        self,
        candidate: QuestionCandidate | None,
    ) -> list[ConversationTurn]:
        recent_turns = list(self._turns)
        if candidate is None or candidate.part_count <= 0:
            return recent_turns

        count = candidate.part_count
        if count > len(recent_turns):
            return recent_turns

        tail = recent_turns[-count:]
        if not all(
            turn.speaker == candidate.speaker
            for turn in tail
        ):
            return recent_turns

        tail_text = " ".join(turn.text for turn in tail).strip()
        if tail_text != candidate.text:
            return recent_turns

        return recent_turns[:-count]

    def _start_answer(
        self,
        question: str,
        force: bool,
        speaker: str = "",
        candidate: QuestionCandidate | None = None,
    ) -> None:
        if not self.running or self.assistant is None:
            return

        if not self._answer_lock.acquire(blocking=False):
            # Keep the newest full assembled candidate instead of only the
            # newest STT fragment. This lets a long conversational question
            # continue growing while a previous analysis/answer is running.
            with self._pending_lock:
                self._queued_answer_text = question.strip()
                self._queued_answer_speaker = speaker
                self._queued_answer_force = force
                self._queued_answer_candidate = candidate

            self._diag(
                "analysis_queued",
                speaker=speaker or "unknown",
                force=force,
                assembly=(
                    candidate.assembly_id
                    if candidate is not None
                    else None
                ),
                revision=(
                    candidate.revision
                    if candidate is not None
                    else None
                ),
            )
            self._emit("on_answer_queued", question.strip())
            return

        recent_turns = self._recent_turns_for_candidate(candidate)
        language = self.language

        self._diag(
            "analysis_started",
            speaker=speaker or "unknown",
            force=force,
            assembly=(
                candidate.assembly_id
                if candidate is not None
                else None
            ),
            revision=(
                candidate.revision
                if candidate is not None
                else None
            ),
            parts=(
                candidate.part_count
                if candidate is not None
                else None
            ),
        )
        self._emit("on_question_candidate", question)

        thread = threading.Thread(
            target=self._answer_worker,
            args=(
                question,
                recent_turns,
                language,
                force,
                speaker,
                candidate,
            ),
            name="interview-answer",
            daemon=True,
        )
        thread.start()

    def _dispatch_queued_answer(self) -> None:
        with self._pending_lock:
            question = self._queued_answer_text
            speaker = self._queued_answer_speaker
            force = self._queued_answer_force
            candidate = self._queued_answer_candidate
            self._queued_answer_text = ""
            self._queued_answer_speaker = ""
            self._queued_answer_force = False
            self._queued_answer_candidate = None

        if not question or not self.running:
            return

        self._start_answer(
            question,
            force=force,
            speaker=speaker,
            candidate=candidate,
        )

    def _update_coding_context(
        self,
        analysis,
        reconstructed: str,
    ) -> CodingContext:
        current = self._coding_context

        if analysis.interview_type != "coding":
            return current

        is_new_problem = (
            analysis.coding_new_problem
            or (
                not current.problem
                and bool(
                    analysis.coding_problem
                    or reconstructed
                )
            )
        )

        problem = (
            analysis.coding_problem.strip()
            or (
                reconstructed.strip()
                if is_new_problem
                else current.problem
            )
        )

        if is_new_problem:
            constraints = analysis.coding_constraints
            last_solution = ""
        else:
            # The contextualizer returns the full currently-active constraint
            # set, so a follow-up can replace an obsolete constraint instead
            # of accumulating contradictions forever.
            constraints = (
                analysis.coding_constraints
                if analysis.coding_constraints
                else current.constraints
            )
            last_solution = current.last_solution

        code_language = current.language or INTERVIEW_CODE_LANGUAGE
        if analysis.coding_language != "unknown":
            code_language = analysis.coding_language

        request = analysis.coding_request
        if request == "none":
            request = current.request

        self._coding_context = CodingContext(
            problem=problem,
            request=request,
            constraints=constraints,
            language=code_language,
            last_solution=last_solution,
        )
        return self._coding_context

    def _answer_worker(
        self,
        question: str,
        recent_turns: list[ConversationTurn],
        language: str,
        force: bool,
        speaker: str,
        candidate: QuestionCandidate | None,
    ) -> None:
        try:
            assistant = self.assistant
            if assistant is None or not self.running:
                return

            analyze_started_at = time.perf_counter()
            analysis = assistant.analyze_turn(
                raw_turn=question,
                recent_turns=recent_turns,
                topic_memory=self._topic_memory,
                coding_context=self._coding_context,
            )
            analyze_ms = (
                time.perf_counter() - analyze_started_at
            ) * 1000.0
            self._emit("on_latency", "analyze", analyze_ms)
            self._diag(
                "latency",
                stage="analyze",
                ms=f"{analyze_ms:.1f}",
                type=analysis.interview_type,
                action=analysis.action,
                assembly=(
                    candidate.assembly_id
                    if candidate is not None
                    else None
                ),
                revision=(
                    candidate.revision
                    if candidate is not None
                    else None
                ),
            )

            # If more speech arrived for the same assembled turn while Groq was
            # analyzing an older revision, that result is obsolete. Do not let
            # it update topic/coding state or produce an answer.
            if not force and self._candidate_is_stale(candidate):
                self._diag(
                    "analysis_stale",
                    assembly=(
                        candidate.assembly_id
                        if candidate is not None
                        else None
                    ),
                    revision=(
                        candidate.revision
                        if candidate is not None
                        else None
                    ),
                )
                self._emit("on_question_waiting", question.strip())
                return

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

            display_topic = analysis.topic
            if analysis.interview_type == "coding":
                display_topic = (
                    f"Coding · {analysis.topic}"
                    if analysis.topic
                    else "Coding"
                )
            elif analysis.interview_type == "behavioral":
                display_topic = (
                    f"Behavioral · {analysis.topic}"
                    if analysis.topic
                    else "Behavioral"
                )

            self._emit(
                "on_turn_understood",
                reconstructed,
                display_topic,
                analysis.language,
            )

            # Behavioral assistance is intentionally outside Stage 2 scope.
            if analysis.interview_type == "behavioral":
                self._discard_candidate(candidate)
                self._diag(
                    "assembly_discarded",
                    reason="behavioral",
                    assembly=(
                        candidate.assembly_id
                        if candidate is not None
                        else None
                    ),
                    revision=(
                        candidate.revision
                        if candidate is not None
                        else None
                    ),
                )
                self._emit("on_question_ignored", reconstructed)
                return

            if not force:
                if analysis.action == "WAIT":
                    # WAIT keeps the complete assembly intact. The next final
                    # STT segment extends the same turn and creates a newer
                    # revision for contextual analysis.
                    self._diag(
                        "assembly_wait",
                        assembly=(
                            candidate.assembly_id
                            if candidate is not None
                            else None
                        ),
                        revision=(
                            candidate.revision
                            if candidate is not None
                            else None
                        ),
                        parts=(
                            candidate.part_count
                            if candidate is not None
                            else None
                        ),
                    )
                    self._emit("on_question_waiting", reconstructed)
                    return

                if analysis.action != "ANSWER":
                    self._discard_candidate(candidate)
                    self._diag(
                        "assembly_discarded",
                        reason=analysis.action.lower(),
                        assembly=(
                            candidate.assembly_id
                            if candidate is not None
                            else None
                        ),
                        revision=(
                            candidate.revision
                            if candidate is not None
                            else None
                        ),
                    )
                    self._emit("on_question_ignored", reconstructed)
                    return

                # Atomically consume only the exact revision that was analyzed.
                # If another segment arrived between the stale check above and
                # this point, the claim fails and this result is discarded.
                if not self._claim_candidate(candidate):
                    self._diag(
                        "analysis_stale",
                        assembly=(
                            candidate.assembly_id
                            if candidate is not None
                            else None
                        ),
                        revision=(
                            candidate.revision
                            if candidate is not None
                            else None
                        ),
                        phase="claim",
                    )
                    self._emit("on_question_waiting", reconstructed)
                    return

            coding_context = self._update_coding_context(
                analysis,
                reconstructed,
            )

            self._last_question = reconstructed
            self._diag(
                "question_confirmed",
                assembly=(
                    candidate.assembly_id
                    if candidate is not None
                    else None
                ),
                revision=(
                    candidate.revision
                    if candidate is not None
                    else None
                ),
                type=analysis.interview_type,
            )
            self._emit("on_question_detected", reconstructed)
            self._emit("on_answer_started", reconstructed)

            generated_answer = ""
            generation_started_at = time.perf_counter()

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
                    interview_type=analysis.interview_type,
                    coding_context=(
                        coding_context
                        if analysis.interview_type == "coding"
                        else None
                    ),
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

            answer_ms = (
                time.perf_counter() - generation_started_at
            ) * 1000.0
            self._emit("on_latency", "answer", answer_ms)
            self._diag(
                "latency",
                stage="answer",
                ms=f"{answer_ms:.1f}",
                type=analysis.interview_type,
                produced=bool(generated_answer),
            )

            if generated_answer:
                self._last_answer = generated_answer

                if analysis.interview_type == "coding":
                    self._coding_context = CodingContext(
                        problem=coding_context.problem,
                        request=coding_context.request,
                        constraints=coding_context.constraints,
                        language=coding_context.language,
                        last_solution=generated_answer,
                    )

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
            self._dispatch_queued_answer()

    def _emit(self, name: str, *args) -> None:
        callback = self.callbacks.get(name)
        if callback:
            callback(*args)
