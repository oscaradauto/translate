"""Local Faster-Whisper transcription service for Stage 1.

The service keeps one Whisper model loaded in memory and serializes inference so
microphone and system-audio captions do not compete for CPU/GPU resources.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

import numpy as np
from faster_whisper import WhisperModel

from config import (
    WHISPER_COMPUTE_TYPE,
    WHISPER_CPU_THREADS,
    WHISPER_DEVICE,
    WHISPER_INITIAL_PROMPT,
    WHISPER_MODEL,
    WHISPER_NUM_WORKERS,
)

PartialCallback = Callable[[str, str, int], None]
FinalCallback = Callable[[str, str, int], None]
StatusCallback = Callable[[str], None]
ErrorCallback = Callable[[str, Exception], None]


class LocalWhisperTranscriber:
    """Shared local transcription engine for all Stage 1 audio sources."""

    def __init__(
        self,
        on_partial: PartialCallback,
        on_final: FinalCallback,
        on_status: StatusCallback | None = None,
        on_error: ErrorCallback | None = None,
        model_name: str = WHISPER_MODEL,
        language: str | None = "en",
        initial_prompt: str | None = WHISPER_INITIAL_PROMPT,
    ) -> None:
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_status = on_status
        self.on_error = on_error
        self.model_name = model_name
        self.language = language
        self.initial_prompt = initial_prompt

        self._model: WhisperModel | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._partial_inflight: set[tuple[str, int]] = set()
        self._partial_pending: dict[tuple[str, int], bytes] = {}
        self._partial_text: dict[tuple[str, int], str] = {}
        self._finalizing: set[tuple[str, int]] = set()
        self._lock = threading.Lock()
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return

        language_label = self.language or "auto"
        self._notify_status(
            f"Cargando Faster-Whisper ({self.model_name}, {language_label})..."
        )

        try:
            self._model = WhisperModel(
                self.model_name,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
                cpu_threads=WHISPER_CPU_THREADS,
                num_workers=WHISPER_NUM_WORKERS,
            )
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="faster-whisper",
            )
            self._running = True
        except Exception as exc:
            self._report_error("LOCAL", exc)
            raise

    def submit_partial(
        self,
        source: str,
        segment_id: int,
        pcm16: bytes,
    ) -> bool:
        """Queue the newest rolling audio window for a live caption.

        There is never more than one queued partial per segment. If Whisper is
        still busy, the older waiting snapshot is replaced by the newest one.
        This "latest wins" policy prevents realtime captions from falling
        further and further behind the meeting.
        """
        if not self._running or not pcm16 or self._executor is None:
            return False

        key = (source, segment_id)
        with self._lock:
            if key in self._finalizing:
                return False

            if key in self._partial_inflight:
                self._partial_pending[key] = pcm16
                return True

            self._partial_inflight.add(key)

        self._launch_partial(key, source, segment_id, pcm16)
        return True

    def _launch_partial(
        self,
        key: tuple[str, int],
        source: str,
        segment_id: int,
        pcm16: bytes,
    ) -> None:
        executor = self._executor
        if executor is None or not self._running:
            with self._lock:
                self._partial_inflight.discard(key)
            return

        future = executor.submit(self._transcribe, pcm16)
        future.add_done_callback(
            lambda f: self._finish_partial(key, source, segment_id, f)
        )

    def submit_final(
        self,
        source: str,
        segment_id: int,
        pcm16: bytes,
    ) -> bool:
        if not self._running or not pcm16 or self._executor is None:
            return False

        key = (source, segment_id)
        with self._lock:
            self._finalizing.add(key)
            self._partial_pending.pop(key, None)

        future = self._executor.submit(self._transcribe, pcm16)
        future.add_done_callback(
            lambda f: self._finish_final(source, segment_id, f)
        )
        return True

    def stop(self) -> None:
        self._running = False

        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

        with self._lock:
            self._partial_inflight.clear()
            self._partial_pending.clear()
            self._partial_text.clear()
            self._finalizing.clear()

        self._model = None

    def _transcribe(self, pcm16: bytes) -> str:
        model = self._model
        if model is None:
            return ""

        audio = (
            np.frombuffer(pcm16, dtype=np.int16)
            .astype(np.float32)
            / 32768.0
        )

        if audio.size == 0:
            return ""

        segments, _ = model.transcribe(
            audio,
            language=self.language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=self.initial_prompt or None,
        )

        parts = [
            segment.text.strip()
            for segment in segments
            if segment.text and segment.text.strip()
        ]
        return " ".join(parts).strip()

    def _finish_partial(
        self,
        key: tuple[str, int],
        source: str,
        segment_id: int,
        future: Future,
    ) -> None:
        try:
            text = future.result().strip()
        except Exception as exc:
            text = ""
            self._report_error(source, exc)

        pending: bytes | None = None
        should_emit = False
        merged = ""

        with self._lock:
            if self._running and text:
                previous = self._partial_text.get(key, "")
                merged = self._merge_incremental_text(previous, text)
                self._partial_text[key] = merged
                should_emit = True

            # If the final transcription has already been requested, do not
            # launch another partial. Otherwise immediately process the newest
            # snapshot that arrived while Whisper was busy.
            if key not in self._finalizing:
                pending = self._partial_pending.pop(key, None)

            if pending is None:
                self._partial_inflight.discard(key)

        if should_emit:
            self.on_partial(source, merged, segment_id)

        if pending is not None and self._running:
            self._launch_partial(key, source, segment_id, pending)

    def _finish_final(
        self,
        source: str,
        segment_id: int,
        future: Future,
    ) -> None:
        key = (source, segment_id)

        try:
            text = future.result().strip()
        except Exception as exc:
            text = ""
            self._report_error(source, exc)

        with self._lock:
            self._partial_pending.pop(key, None)
            self._partial_inflight.discard(key)
            self._partial_text.pop(key, None)
            self._finalizing.discard(key)

        if self._running and text:
            self.on_final(source, text, segment_id)

    @staticmethod
    def _merge_incremental_text(previous: str, current: str) -> str:
        """Merge overlapping rolling-window transcripts into one live caption."""
        previous = previous.strip()
        current = current.strip()

        if not previous:
            return current
        if not current:
            return previous

        if current.casefold().startswith(previous.casefold()):
            return current
        if current.casefold() in previous.casefold():
            return previous

        prev_words = previous.split()
        curr_words = current.split()

        def normalized(word: str) -> str:
            folded = unicodedata.normalize("NFD", word.casefold())
            folded = "".join(
                char
                for char in folded
                if unicodedata.category(char) != "Mn"
            )
            return re.sub(r"[^a-z0-9]+", "", folded)

        max_overlap = min(14, len(prev_words), len(curr_words))
        for count in range(max_overlap, 0, -1):
            left = [normalized(w) for w in prev_words[-count:]]
            right = [normalized(w) for w in curr_words[:count]]
            if left == right and any(left):
                suffix = " ".join(curr_words[count:])
                return previous if not suffix else f"{previous} {suffix}"

        # Whisper occasionally rewrites the rolling window enough that there is
        # no exact overlap. Keeping both fragments is preferable to replacing
        # already-visible text and making the caption jump backwards.
        return f"{previous} {current}".strip()

    def _notify_status(self, text: str) -> None:
        if self.on_status:
            self.on_status(text)

    def _report_error(self, source: str, exc: Exception) -> None:
        if self.on_error:
            self.on_error(source, exc)
