"""Audio capture and orchestration for Stage 1 English live subtitles.

MICROPHONE -> YOU
SYSTEM AUDIO -> COMPANION

Stage 1 intentionally contains no translation and no interview-agent logic.
Audio is captured locally and streamed to the specialized OpenAI realtime
speech-to-text engine.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pyaudio
import soundcard as sc
import webrtcvad
from scipy.signal import resample_poly

from config import (
    MIC_DEVICE_INDEX,
    OPENAI_TRANSCRIPTION_DELAY,
    OPENAI_TRANSCRIPTION_KEYWORDS,
    OPENAI_TRANSCRIPTION_MODEL,
    OPENAI_TRANSCRIPTION_PROMPT,
)
from realtime_transcriber import RealtimeTranscriptionSession

TARGET_SAMPLE_RATE = 24000
PREFERRED_MIC_SAMPLE_RATE = 48000
FALLBACK_MIC_SAMPLE_RATE = 16000
SYSTEM_SAMPLE_RATE = 48000
VAD_FRAME_MS = 30
VAD_MODE = 2
SPEECH_END_SILENCE_MS = 600


def _pcm16_to_float32(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0


def _float32_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def _resample_to_target(audio: np.ndarray, source_rate: int) -> bytes:
    if audio.size == 0:
        return b""

    if source_rate == TARGET_SAMPLE_RATE:
        return _float32_to_pcm16(audio)

    # resample_poly is efficient for the short 30 ms blocks used by live STT.
    gcd = np.gcd(source_rate, TARGET_SAMPLE_RATE)
    up = TARGET_SAMPLE_RATE // gcd
    down = source_rate // gcd
    resampled = resample_poly(audio, up, down)
    return _float32_to_pcm16(resampled.astype(np.float32))


def _is_speech(pcm16: bytes, sample_rate: int, vad: webrtcvad.Vad) -> bool:
    frame_bytes = int(sample_rate * VAD_FRAME_MS / 1000) * 2
    if len(pcm16) != frame_bytes:
        return False
    try:
        return vad.is_speech(pcm16, sample_rate)
    except Exception:
        return False


class RealtimeMicStreamer:
    """Captura el micrófono físico y lo envía a gpt-live-transcribe."""

    def __init__(self, device_index, on_partial, on_final, on_error):
        self.device_index = device_index
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_error = on_error

        self.running = False
        self.thread: threading.Thread | None = None
        self.audio = None
        self.stream = None
        self.sample_rate = None
        self.session: RealtimeTranscriptionSession | None = None

    def start(self) -> None:
        if self.running:
            return

        self.session = RealtimeTranscriptionSession(
            source="YOU",
            on_partial=self.on_partial,
            on_final=self.on_final,
            on_status=lambda source, status: None,
            on_error=self._on_session_error,
            model=OPENAI_TRANSCRIPTION_MODEL,
            delay=OPENAI_TRANSCRIPTION_DELAY,
            prompt=OPENAI_TRANSCRIPTION_PROMPT,
            keywords=OPENAI_TRANSCRIPTION_KEYWORDS,
        )
        self.session.start()

        self.audio = pyaudio.PyAudio()
        self.stream = self._open_microphone_stream()

        self.running = True
        self.thread = threading.Thread(
            target=self._capture_loop,
            name="microphone-capture",
            daemon=True,
        )
        self.thread.start()

    def _open_microphone_stream(self):
        errors = []

        for sample_rate in (PREFERRED_MIC_SAMPLE_RATE, FALLBACK_MIC_SAMPLE_RATE):
            frames = int(sample_rate * VAD_FRAME_MS / 1000)
            try:
                stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=frames,
                )
                self.sample_rate = sample_rate
                return stream
            except Exception as exc:
                errors.append(str(exc))

        raise RuntimeError(
            "No se pudo abrir el micrófono. "
            + " | ".join(errors[-2:])
        )

    def _capture_loop(self) -> None:
        vad = webrtcvad.Vad(VAD_MODE)
        speech_active = False
        last_speech_time = 0.0

        while self.running:
            try:
                pcm16 = self.stream.read(
                    int(self.sample_rate * VAD_FRAME_MS / 1000),
                    exception_on_overflow=False,
                )
            except Exception as exc:
                if self.running:
                    self._on_capture_error(exc)
                break

            now = time.monotonic()
            is_speech = _is_speech(pcm16, self.sample_rate, vad)

            if self.session and self.session.append_audio(
                _resample_to_target(
                    _pcm16_to_float32(pcm16),
                    self.sample_rate,
                )
            ):
                if is_speech:
                    speech_active = True
                    last_speech_time = now
                elif speech_active and (
                    (now - last_speech_time) * 1000 >= SPEECH_END_SILENCE_MS
                ):
                    self.session.commit()
                    speech_active = False

    def _on_capture_error(self, exc: Exception) -> None:
        if self.on_error:
            self.on_error("YOU", exc)

    def _on_session_error(self, source: str, exc: Exception) -> None:
        if self.on_error:
            self.on_error(source, exc)

    def stop(self) -> None:
        self.running = False

        if self.stream is not None:
            try:
                self.stream.stop_stream()
            except Exception:
                pass
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None

        if self.session:
            self.session.stop()
            self.session = None

        if self.audio:
            try:
                self.audio.terminate()
            except Exception:
                pass
            self.audio = None


class RealtimeSystemAudioStreamer:
    """Captura el audio del sistema/Teams/Zoom/Meet y lo transcribe."""

    def __init__(self, on_partial, on_final, on_error):
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_error = on_error

        self.running = False
        self.thread: threading.Thread | None = None
        self.session: RealtimeTranscriptionSession | None = None

    def start(self) -> None:
        if self.running:
            return

        self.session = RealtimeTranscriptionSession(
            source="COMPANION",
            on_partial=self.on_partial,
            on_final=self.on_final,
            on_status=lambda source, status: None,
            on_error=self._on_session_error,
            model=OPENAI_TRANSCRIPTION_MODEL,
            delay=OPENAI_TRANSCRIPTION_DELAY,
            prompt=OPENAI_TRANSCRIPTION_PROMPT,
            keywords=OPENAI_TRANSCRIPTION_KEYWORDS,
        )
        self.session.start()

        self.running = True
        self.thread = threading.Thread(
            target=self._capture_loop,
            name="system-audio-capture",
            daemon=True,
        )
        self.thread.start()

    def _capture_loop(self) -> None:
        vad = webrtcvad.Vad(VAD_MODE)
        speech_active = False
        last_speech_time = 0.0

        try:
            speaker = sc.default_speaker()
            loopback_mic = sc.get_microphone(
                speaker.name,
                include_loopback=True,
            )

            with loopback_mic.recorder(
                samplerate=SYSTEM_SAMPLE_RATE
            ) as recorder:
                while self.running:
                    data = recorder.record(
                        numframes=int(
                            SYSTEM_SAMPLE_RATE * VAD_FRAME_MS / 1000
                        )
                    )
                    mono = data.mean(axis=1) if data.ndim > 1 else data
                    pcm16 = _float32_to_pcm16(mono)
                    is_speech = _is_speech(
                        pcm16,
                        SYSTEM_SAMPLE_RATE,
                        vad,
                    )

                    if self.session and self.session.append_audio(
                        _resample_to_target(mono, SYSTEM_SAMPLE_RATE)
                    ):
                        now = time.monotonic()
                        if is_speech:
                            speech_active = True
                            last_speech_time = now
                        elif speech_active and (
                            (now - last_speech_time) * 1000
                            >= SPEECH_END_SILENCE_MS
                        ):
                            self.session.commit()
                            speech_active = False

        except Exception as exc:
            if self.running:
                self._on_capture_error(exc)

    def _on_capture_error(self, exc: Exception) -> None:
        if self.on_error:
            self.on_error("COMPANION", exc)

    def _on_session_error(self, source: str, exc: Exception) -> None:
        if self.on_error:
            self.on_error(source, exc)

    def stop(self) -> None:
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None

        if self.session:
            self.session.stop()
            self.session = None


class ListenerController:
    """Coordina las dos fuentes de audio de Stage 1."""

    def __init__(self, callbacks, mic_device=MIC_DEVICE_INDEX):
        self.callbacks = callbacks
        self.mic_device = mic_device
        self.mic_streamer: RealtimeMicStreamer | None = None
        self.loopback_streamer: RealtimeSystemAudioStreamer | None = None
        self.running = False

        self._segment_counter = 0
        self._counter_lock = threading.Lock()

    def _next_segment_id(self) -> int:
        with self._counter_lock:
            self._segment_counter += 1
            return self._segment_counter

    def _on_partial(self, source: str, text: str, item_id: str) -> None:
        if self.running and text:
            self._emit("on_subtitle_partial", source, text)

    def _on_final(self, source: str, text: str, item_id: str) -> None:
        if not self.running or len(text.strip()) < 2:
            return

        segment_id = self._next_segment_id()
        self._emit("on_subtitle", source, text.strip(), segment_id)

    def start(self) -> None:
        if self.running:
            return

        self.running = True
        self._emit("on_status", "Preparando audio...")

        started = 0

        try:
            self.mic_streamer = RealtimeMicStreamer(
                self.mic_device,
                self._on_partial,
                self._on_final,
                self._on_source_error,
            )
            self.mic_streamer.start()
            started += 1
        except Exception as exc:
            self._on_source_error("YOU", exc)

        try:
            self.loopback_streamer = RealtimeSystemAudioStreamer(
                self._on_partial,
                self._on_final,
                self._on_source_error,
            )
            self.loopback_streamer.start()
            started += 1
        except Exception as exc:
            self._on_source_error("COMPANION", exc)

        if started:
            self._emit("on_status", "Escuchando...")
        else:
            self.running = False
            self._emit("on_status", "Error de audio")

    def _on_source_error(self, source: str, exc: Exception) -> None:
        print(f"[{source}] {exc}")

        mic_alive = bool(
            self.mic_streamer
            and self.mic_streamer.session
            and self.mic_streamer.session.is_running
        )
        companion_alive = bool(
            self.loopback_streamer
            and self.loopback_streamer.session
            and self.loopback_streamer.session.is_running
        )

        if mic_alive and not companion_alive:
            self._emit("on_status", "Escuchando (solo YOU)")
        elif companion_alive and not mic_alive:
            self._emit("on_status", "Escuchando (solo COMPANION)")
        elif not mic_alive and not companion_alive:
            self._emit("on_status", "Error de audio")

    def stop(self) -> None:
        if not self.running and not self.mic_streamer and not self.loopback_streamer:
            return

        self.running = False

        if self.mic_streamer:
            self.mic_streamer.stop()
            self.mic_streamer = None

        if self.loopback_streamer:
            self.loopback_streamer.stop()
            self.loopback_streamer = None

        self._emit("on_status", "Detenido")

    def _emit(self, callback_name, *args):
        callback = self.callbacks.get(callback_name)
        if callback:
            callback(*args)
