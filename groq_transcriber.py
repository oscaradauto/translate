"""Groq speech-to-text adapter for Stage 2.

Receives 16 kHz mono PCM16 utterances from the existing VAD/capture pipeline
and submits only finalized interview turns to Groq Whisper.
"""

from __future__ import annotations

import io
import os
import threading
import time
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

from groq import Groq

from config import (
    GROQ_STT_MODEL,
    GROQ_STT_PROMPT_AUTO,
    GROQ_STT_PROMPT_EN,
    GROQ_STT_PROMPT_ES,
)

PartialCallback = Callable[[str, str, int], None]
FinalCallback = Callable[[str, str, int], None]
StatusCallback = Callable[[str], None]
ErrorCallback = Callable[[str, Exception], None]


class GroqSpeechTranscriber:
    """Final-utterance STT optimized for interview-question accuracy."""

    def __init__(
        self,
        on_partial: PartialCallback,
        on_final: FinalCallback,
        on_status: StatusCallback | None = None,
        on_error: ErrorCallback | None = None,
        language: str | None = None,
    ) -> None:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY no está configurada en el archivo .env."
            )

        self.on_partial = on_partial
        self.on_final = on_final
        self.on_status = on_status
        self.on_error = on_error
        self.language = language

        self.client = Groq(api_key=api_key)
        self._executor: ThreadPoolExecutor | None = None
        self._running = False
        self._lock = threading.Lock()
        self._futures: set[Future] = set()

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return

        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="groq-stt",
        )
        self._running = True

        if self.on_status:
            self.on_status(
                f"Groq STT listo ({GROQ_STT_MODEL})"
            )

    def submit_partial(
        self,
        source: str,
        segment_id: int,
        pcm16: bytes,
    ) -> bool:
        # Stage 2 intentionally favors accurate finalized turns over noisy
        # rolling local partials. The question appears after a natural pause.
        return False

    def submit_final(
        self,
        source: str,
        segment_id: int,
        pcm16: bytes,
    ) -> bool:
        executor = self._executor
        if not self._running or executor is None or not pcm16:
            return False

        started_at = time.perf_counter()
        future = executor.submit(self._transcribe, pcm16)
        with self._lock:
            self._futures.add(future)

        future.add_done_callback(
            lambda f, started=started_at: self._finish_final(
                source,
                segment_id,
                f,
                started,
            )
        )
        return True

    def stop(self) -> None:
        self._running = False

        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

        with self._lock:
            self._futures.clear()

    def _transcribe(self, pcm16: bytes) -> str:
        wav_bytes = self._to_wav(pcm16)

        kwargs = {
            "file": ("interview.wav", wav_bytes),
            "model": GROQ_STT_MODEL,
            "response_format": "json",
            "temperature": 0.0,
        }

        prompt = self._prompt_for_language()
        if prompt:
            kwargs["prompt"] = prompt

        if self.language:
            kwargs["language"] = self.language

        transcription = self.client.audio.transcriptions.create(
            **kwargs
        )
        return (transcription.text or "").strip()

    def _prompt_for_language(self) -> str:
        if self.language == "es":
            return GROQ_STT_PROMPT_ES.strip()
        if self.language == "en":
            return GROQ_STT_PROMPT_EN.strip()
        return GROQ_STT_PROMPT_AUTO.strip()

    @staticmethod
    def _to_wav(pcm16: bytes) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(pcm16)
        return buffer.getvalue()

    def _finish_final(
        self,
        source: str,
        segment_id: int,
        future: Future,
        started_at: float,
    ) -> None:
        with self._lock:
            self._futures.discard(future)

        if not self._running:
            return

        try:
            text = future.result().strip()
        except Exception as exc:
            if self.on_error:
                self.on_error(source, exc)
            return

        latency_ms = (time.perf_counter() - started_at) * 1000.0
        if self.on_status:
            self.on_status(
                f"__STAGE2_LATENCY__:stt:{latency_ms:.1f}"
            )

        if text:
            self.on_final(source, text, segment_id)
