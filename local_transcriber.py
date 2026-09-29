"""Local Faster-Whisper transcription service for Stage 1.

The service keeps one Whisper model loaded in memory and serializes inference so
microphone and system-audio captions do not compete for CPU/GPU resources.
"""

from __future__ import annotations

import threading
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
    ) -> None:
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_status = on_status
        self.on_error = on_error

        self._model: WhisperModel | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._partial_inflight: set[tuple[str, int]] = set()
        self._lock = threading.Lock()
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return

        self._notify_status(
            f"Cargando Faster-Whisper ({WHISPER_MODEL})..."
        )

        try:
            self._model = WhisperModel(
                WHISPER_MODEL,
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
        """Transcribe a snapshot of the current utterance.

        Only one partial inference per source/segment is allowed at a time. This
        prevents a slow CPU from building a backlog of stale partial captions.
        """
        if not self._running or not pcm16 or self._executor is None:
            return False

        key = (source, segment_id)
        with self._lock:
            if key in self._partial_inflight:
                return False
            self._partial_inflight.add(key)

        future = self._executor.submit(self._transcribe, pcm16)
        future.add_done_callback(
            lambda f: self._finish_partial(key, source, segment_id, f)
        )
        return True

    def submit_final(
        self,
        source: str,
        segment_id: int,
        pcm16: bytes,
    ) -> bool:
        if not self._running or not pcm16 or self._executor is None:
            return False

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
            language="en",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=WHISPER_INITIAL_PROMPT,
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
        with self._lock:
            self._partial_inflight.discard(key)

        if not self._running:
            return

        try:
            text = future.result().strip()
        except Exception as exc:
            self._report_error(source, exc)
            return

        if text:
            self.on_partial(source, text, segment_id)

    def _finish_final(
        self,
        source: str,
        segment_id: int,
        future: Future,
    ) -> None:
        if not self._running:
            return

        try:
            text = future.result().strip()
        except Exception as exc:
            self._report_error(source, exc)
            return

        if text:
            self.on_final(source, text, segment_id)

    def _notify_status(self, text: str) -> None:
        if self.on_status:
            self.on_status(text)

    def _report_error(self, source: str, exc: Exception) -> None:
        if self.on_error:
            self.on_error(source, exc)
