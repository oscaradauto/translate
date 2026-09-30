"""Audio capture and orchestration for Stage 1 live English subtitles.

Stage 1 is intentionally local and contains no assistant logic:
MICROPHONE -> YOU
SYSTEM AUDIO -> MEETING
AUDIO -> WebRTC VAD -> Faster-Whisper -> English captions
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

import numpy as np
import pyaudio
import soundcard as sc
import webrtcvad
from scipy.signal import resample_poly

from config import (
    LOCAL_SPEECH_GATE_HOLD_MS,
    MEETING_MIN_VOICED_MS,
    MEETING_SPEECH_START_MS,
    MEETING_VAD_MODE,
    MIC_DEVICE_INDEX,
    MIC_MIN_DBFS,
    MIC_MIN_VOICED_MS,
    MIC_SPEECH_START_MS,
    MIC_VAD_MODE,
    SUBTITLE_MAX_UTTERANCE_SECONDS,
    SUBTITLE_PARTIAL_INTERVAL_SECONDS,
    SUBTITLE_PARTIAL_MIN_SECONDS,
    SUBTITLE_SPEECH_END_MS,
    WHISPER_PARTIAL_WINDOW_SECONDS,
    WHISPER_INITIAL_PROMPT,
    WHISPER_MODEL,
)
from local_transcriber import LocalWhisperTranscriber

TARGET_SAMPLE_RATE = 16000
PREFERRED_MIC_SAMPLE_RATE = 48000
FALLBACK_MIC_SAMPLE_RATE = 16000
SYSTEM_SAMPLE_RATE = 48000
VAD_FRAME_MS = 30
PRE_ROLL_MS = 240

_PRE_ROLL_FRAMES = max(1, PRE_ROLL_MS // VAD_FRAME_MS)


class _LocalSpeechGate:
    """Coordinates source ownership between microphone and meeting loopback.

    When the physical microphone detects local speech, YOU wins for a short
    interval. The meeting loopback is suppressed during that interval so a
    locally spoken sentence cannot also be emitted as MEETING.
    """

    def __init__(self, hold_ms: int) -> None:
        self._hold_seconds = max(0.0, hold_ms / 1000.0)
        self._active_until = 0.0
        self._lock = threading.Lock()

    def mark_local_speech(self, now: float) -> None:
        with self._lock:
            self._active_until = max(
                self._active_until,
                now + self._hold_seconds,
            )

    def is_local_speech_active(self, now: float) -> bool:
        with self._lock:
            return now <= self._active_until


def _pcm16_to_float32(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0


def _float32_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def _resample_pcm16(pcm16: bytes, source_rate: int) -> bytes:
    if not pcm16:
        return b""
    if source_rate == TARGET_SAMPLE_RATE:
        return pcm16

    audio = _pcm16_to_float32(pcm16)
    gcd = int(np.gcd(source_rate, TARGET_SAMPLE_RATE))
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


def _dbfs(pcm16: bytes) -> float:
    """Return RMS level in dBFS for a PCM16 mono frame."""
    if not pcm16:
        return -96.0

    samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return -96.0

    rms = float(np.sqrt(np.mean(samples * samples)))
    if rms <= 1.0:
        return -96.0

    return 20.0 * np.log10(rms / 32768.0)


class _SpeechSegmenter:
    """Builds stable utterances and periodically requests live partial captions."""

    def __init__(
        self,
        source: str,
        transcriber: LocalWhisperTranscriber,
        next_segment_id: Callable[[], int],
        speech_start_ms: int,
        min_voiced_ms: int,
        speech_end_ms: int = SUBTITLE_SPEECH_END_MS,
        max_utterance_seconds: float = SUBTITLE_MAX_UTTERANCE_SECONDS,
        partials_enabled: bool = True,
    ) -> None:
        self.source = source
        self.transcriber = transcriber
        self.next_segment_id = next_segment_id
        self._speech_end_ms = speech_end_ms
        self._max_utterance_seconds = max_utterance_seconds
        self._partials_enabled = partials_enabled
        self._speech_start_frames = max(
            1,
            int(np.ceil(speech_start_ms / VAD_FRAME_MS)),
        )
        self._min_voiced_frames = max(
            1,
            int(np.ceil(min_voiced_ms / VAD_FRAME_MS)),
        )

        self._pre_roll: deque[bytes] = deque(maxlen=_PRE_ROLL_FRAMES)
        self._frames: list[bytes] = []
        self._speech_active = False
        self._segment_id: int | None = None
        self._last_speech_time = 0.0
        self._last_partial_time = 0.0
        self._speech_candidate_frames = 0
        self._voiced_frames = 0

    def process(self, pcm16_target: bytes, is_speech: bool, now: float) -> None:
        if not pcm16_target:
            return

        if not self._speech_active:
            self._pre_roll.append(pcm16_target)

            if is_speech:
                self._speech_candidate_frames += 1
            else:
                self._speech_candidate_frames = 0
                return

            # One noisy frame is not enough to create a caption. Requiring
            # consecutive voiced frames prevents false YOU segments.
            if self._speech_candidate_frames < self._speech_start_frames:
                return

            self._speech_active = True
            self._segment_id = self.next_segment_id()
            self._frames = list(self._pre_roll)
            self._pre_roll.clear()
            self._voiced_frames = self._speech_candidate_frames
            self._speech_candidate_frames = 0
            self._last_speech_time = now
            self._last_partial_time = now
            return

        self._frames.append(pcm16_target)

        if is_speech:
            self._voiced_frames += 1
            self._last_speech_time = now

        duration_seconds = (
            len(self._frames) * VAD_FRAME_MS / 1000.0
        )

        if (
            self._partials_enabled
            and duration_seconds >= SUBTITLE_PARTIAL_MIN_SECONDS
            and self._voiced_frames >= self._min_voiced_frames
            and now - self._last_partial_time
            >= SUBTITLE_PARTIAL_INTERVAL_SECONDS
        ):
            self._last_partial_time = now
            self._submit_partial()

        silence_ms = (now - self._last_speech_time) * 1000
        if silence_ms >= self._speech_end_ms:
            self._finalize()
        elif duration_seconds >= self._max_utterance_seconds:
            # Long monologues are split into readable caption blocks.
            self._finalize()

    def flush(self) -> None:
        if self._speech_active and self._frames:
            self._finalize()

    def abort(self) -> None:
        """Drop the current candidate/utterance without transcribing it."""
        self._speech_active = False
        self._segment_id = None
        self._frames = []
        self._last_speech_time = 0.0
        self._last_partial_time = 0.0
        self._speech_candidate_frames = 0
        self._voiced_frames = 0
        self._pre_roll.clear()

    def _submit_partial(self) -> None:
        if self._segment_id is None or not self._frames:
            return

        window_frames = max(
            1,
            int(
                np.ceil(
                    WHISPER_PARTIAL_WINDOW_SECONDS
                    * 1000
                    / VAD_FRAME_MS
                )
            ),
        )
        rolling_frames = self._frames[-window_frames:]

        self.transcriber.submit_partial(
            self.source,
            self._segment_id,
            b"".join(rolling_frames),
        )

    def _finalize(self) -> None:
        segment_id = self._segment_id
        frames = self._frames
        voiced_frames = self._voiced_frames

        self._speech_active = False
        self._segment_id = None
        self._frames = []
        self._last_speech_time = 0.0
        self._last_partial_time = 0.0
        self._speech_candidate_frames = 0
        self._voiced_frames = 0
        self._pre_roll.clear()

        if segment_id is None or not frames:
            return

        # Do not send tiny noise bursts to Whisper. Those bursts are a common
        # source of hallucinations such as "So," or "Now," on silent mics.
        if voiced_frames < self._min_voiced_frames:
            return

        self.transcriber.submit_final(
            self.source,
            segment_id,
            b"".join(frames),
        )


class LocalMicStreamer:
    """Captures the physical microphone and feeds the local STT engine."""

    def __init__(
        self,
        device_index: int,
        transcriber: LocalWhisperTranscriber,
        next_segment_id: Callable[[], int],
        local_speech_gate: _LocalSpeechGate,
        on_error,
        speech_end_ms: int = SUBTITLE_SPEECH_END_MS,
        max_utterance_seconds: float = SUBTITLE_MAX_UTTERANCE_SECONDS,
        partials_enabled: bool = True,
    ) -> None:
        self.device_index = device_index
        self.transcriber = transcriber
        self.next_segment_id = next_segment_id
        self.local_speech_gate = local_speech_gate
        self.on_error = on_error
        self.speech_end_ms = speech_end_ms
        self.max_utterance_seconds = max_utterance_seconds
        self.partials_enabled = partials_enabled

        self.running = False
        self.thread: threading.Thread | None = None
        self.audio: pyaudio.PyAudio | None = None
        self.stream = None
        self.sample_rate: int | None = None

    def start(self) -> None:
        if self.running:
            return

        self.audio = pyaudio.PyAudio()
        try:
            self.stream = self._open_microphone_stream()
        except Exception:
            self.audio.terminate()
            self.audio = None
            raise

        self.running = True
        self.thread = threading.Thread(
            target=self._capture_loop,
            name="microphone-capture",
            daemon=True,
        )
        self.thread.start()

    def _open_microphone_stream(self):
        errors: list[str] = []

        device_index = self.device_index
        if device_index < 0:
            try:
                device_index = int(
                    self.audio.get_default_input_device_info()["index"]
                )
            except Exception as exc:
                raise RuntimeError(
                    f"No se pudo detectar el micrófono predeterminado: {exc}"
                ) from exc

        try:
            device_info = self.audio.get_device_info_by_index(device_index)
            device_name = device_info.get("name", f"device {device_index}")
            max_inputs = int(device_info.get("maxInputChannels", 0))
            if max_inputs < 1:
                raise RuntimeError(
                    f"El dispositivo #{device_index} no es un dispositivo de entrada."
                )
            print(f"[Audio:YOU] Micrófono #{device_index}: {device_name}")
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo validar el micrófono #{device_index}: {exc}"
            ) from exc

        self.device_index = device_index

        for sample_rate in (
            PREFERRED_MIC_SAMPLE_RATE,
            FALLBACK_MIC_SAMPLE_RATE,
        ):
            frames = int(sample_rate * VAD_FRAME_MS / 1000)
            try:
                stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    input_device_index=device_index,
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
        assert self.sample_rate is not None
        vad = webrtcvad.Vad(MIC_VAD_MODE)
        segmenter = _SpeechSegmenter(
            "YOU",
            self.transcriber,
            self.next_segment_id,
            speech_start_ms=MIC_SPEECH_START_MS,
            min_voiced_ms=MIC_MIN_VOICED_MS,
            speech_end_ms=self.speech_end_ms,
            max_utterance_seconds=self.max_utterance_seconds,
            partials_enabled=self.partials_enabled,
        )
        frame_count = int(self.sample_rate * VAD_FRAME_MS / 1000)

        try:
            while self.running:
                pcm16 = self.stream.read(
                    frame_count,
                    exception_on_overflow=False,
                )
                now = time.monotonic()
                target_pcm16 = _resample_pcm16(
                    pcm16,
                    self.sample_rate,
                )
                speech = (
                    _is_speech(target_pcm16, TARGET_SAMPLE_RATE, vad)
                    and _dbfs(target_pcm16) >= MIC_MIN_DBFS
                )

                if speech:
                    # The physical mic is the source of truth for YOU.
                    self.local_speech_gate.mark_local_speech(now)

                segmenter.process(target_pcm16, speech, now)
        except Exception as exc:
            if self.running:
                self.on_error("YOU", exc)
        finally:
            segmenter.flush()

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

        if self.audio is not None:
            try:
                self.audio.terminate()
            except Exception:
                pass
            self.audio = None


class LocalSystemAudioStreamer:
    """Captures Windows speaker loopback audio for the remote meeting."""

    def __init__(
        self,
        transcriber: LocalWhisperTranscriber,
        next_segment_id: Callable[[], int],
        local_speech_gate: _LocalSpeechGate,
        on_error,
        speech_end_ms: int = SUBTITLE_SPEECH_END_MS,
        max_utterance_seconds: float = SUBTITLE_MAX_UTTERANCE_SECONDS,
        partials_enabled: bool = True,
    ) -> None:
        self.transcriber = transcriber
        self.next_segment_id = next_segment_id
        self.local_speech_gate = local_speech_gate
        self.on_error = on_error
        self.speech_end_ms = speech_end_ms
        self.max_utterance_seconds = max_utterance_seconds
        self.partials_enabled = partials_enabled
        self.running = False
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.running:
            return

        # Validate a loopback source before reporting the streamer as started.
        speaker = sc.default_speaker()
        sc.get_microphone(speaker.name, include_loopback=True)

        self.running = True
        self.thread = threading.Thread(
            target=self._capture_loop,
            name="system-audio-capture",
            daemon=True,
        )
        self.thread.start()

    def _capture_loop(self) -> None:
        vad = webrtcvad.Vad(MEETING_VAD_MODE)
        segmenter = _SpeechSegmenter(
            "MEETING",
            self.transcriber,
            self.next_segment_id,
            speech_start_ms=MEETING_SPEECH_START_MS,
            min_voiced_ms=MEETING_MIN_VOICED_MS,
            speech_end_ms=self.speech_end_ms,
            max_utterance_seconds=self.max_utterance_seconds,
            partials_enabled=self.partials_enabled,
        )

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
                            SYSTEM_SAMPLE_RATE
                            * VAD_FRAME_MS
                            / 1000
                        )
                    )
                    mono = (
                        data.mean(axis=1)
                        if data.ndim > 1
                        else data
                    )
                    pcm16 = _float32_to_pcm16(mono)
                    target_pcm16 = _resample_pcm16(
                        pcm16,
                        SYSTEM_SAMPLE_RATE,
                    )
                    now = time.monotonic()

                    # If the physical microphone is currently detecting the
                    # local user, YOU owns this time window. Drop loopback
                    # audio instead of allowing the same voice to be labelled
                    # as a remote meeting participant.
                    if self.local_speech_gate.is_local_speech_active(now):
                        segmenter.abort()
                        continue

                    speech = _is_speech(
                        target_pcm16,
                        TARGET_SAMPLE_RATE,
                        vad,
                    )
                    segmenter.process(
                        target_pcm16,
                        speech,
                        now,
                    )
        except Exception as exc:
            if self.running:
                self.on_error("MEETING", exc)
        finally:
            segmenter.flush()

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None


class ListenerController:
    """Coordinates local transcription for microphone and system audio."""

    def __init__(
        self,
        callbacks,
        mic_device: int = MIC_DEVICE_INDEX,
        whisper_model: str = WHISPER_MODEL,
        transcription_language: str | None = "en",
        initial_prompt: str | None = WHISPER_INITIAL_PROMPT,
        transcriber_factory=None,
        speech_end_ms: int = SUBTITLE_SPEECH_END_MS,
        max_utterance_seconds: float = SUBTITLE_MAX_UTTERANCE_SECONDS,
        partials_enabled: bool = True,
        engine_label: str = "Faster-Whisper local",
    ):
        self.callbacks = callbacks
        self.mic_device = mic_device
        self.whisper_model = whisper_model
        self.transcription_language = transcription_language
        self.initial_prompt = initial_prompt
        self.transcriber_factory = transcriber_factory
        self.speech_end_ms = speech_end_ms
        self.max_utterance_seconds = max_utterance_seconds
        self.partials_enabled = partials_enabled
        self.engine_label = engine_label

        self.running = False
        self.mic_streamer: LocalMicStreamer | None = None
        self.loopback_streamer: LocalSystemAudioStreamer | None = None
        self.transcriber: LocalWhisperTranscriber | None = None

        self._segment_counter = 0
        self._counter_lock = threading.Lock()
        self._local_speech_gate = _LocalSpeechGate(
            LOCAL_SPEECH_GATE_HOLD_MS
        )

    def _next_segment_id(self) -> int:
        with self._counter_lock:
            self._segment_counter += 1
            return self._segment_counter

    def _on_partial(
        self,
        source: str,
        text: str,
        segment_id: int,
    ) -> None:
        if self.running and text:
            self._emit(
                "on_subtitle_partial",
                source,
                text,
                segment_id,
            )

    def _on_final(
        self,
        source: str,
        text: str,
        segment_id: int,
    ) -> None:
        if self.running and len(text.strip()) >= 2:
            self._emit(
                "on_subtitle",
                source,
                text.strip(),
                segment_id,
            )

    def _on_transcription_error(
        self,
        source: str,
        exc: Exception,
    ) -> None:
        print(f"[Transcription:{source}] {exc}")
        self._emit(
            "on_transcription_error",
            source,
            exc,
        )

    def _on_capture_error(self, source: str, exc: Exception) -> None:
        print(f"[Audio:{source}] {exc}")

    def start(self) -> None:
        if self.running:
            return

        self.running = True
        self._emit("on_status", "Cargando modelo local...")

        try:
            if self.transcriber_factory is not None:
                self.transcriber = self.transcriber_factory(
                    on_partial=self._on_partial,
                    on_final=self._on_final,
                    on_status=lambda text: self._emit(
                        "on_status",
                        text,
                    ),
                    on_error=self._on_transcription_error,
                )
            else:
                self.transcriber = LocalWhisperTranscriber(
                    on_partial=self._on_partial,
                    on_final=self._on_final,
                    on_status=lambda text: self._emit(
                        "on_status",
                        text,
                    ),
                    on_error=self._on_transcription_error,
                    model_name=self.whisper_model,
                    language=self.transcription_language,
                    initial_prompt=self.initial_prompt,
                )

            self.transcriber.start()
        except Exception as exc:
            self.running = False
            self._emit(
                "on_status",
                f"Error cargando transcripción: {exc}",
            )
            return

        started_sources: list[str] = []

        try:
            self.mic_streamer = LocalMicStreamer(
                self.mic_device,
                self.transcriber,
                self._next_segment_id,
                self._local_speech_gate,
                self._on_capture_error,
                speech_end_ms=self.speech_end_ms,
                max_utterance_seconds=self.max_utterance_seconds,
                partials_enabled=self.partials_enabled,
            )
            self.mic_streamer.start()
            started_sources.append("YOU")
        except Exception as exc:
            print(f"[Audio:YOU] {exc}")
            self.mic_streamer = None

        try:
            self.loopback_streamer = LocalSystemAudioStreamer(
                self.transcriber,
                self._next_segment_id,
                self._local_speech_gate,
                self._on_capture_error,
                speech_end_ms=self.speech_end_ms,
                max_utterance_seconds=self.max_utterance_seconds,
                partials_enabled=self.partials_enabled,
            )
            self.loopback_streamer.start()
            started_sources.append("MEETING")
        except Exception as exc:
            print(f"[Audio:MEETING] {exc}")
            self.loopback_streamer = None

        if not started_sources:
            self.running = False
            if self.transcriber:
                self.transcriber.stop()
                self.transcriber = None
            self._emit(
                "on_status",
                "Error: no se pudo abrir ninguna fuente de audio",
            )
            return

        if started_sources == ["YOU"]:
            status = "Escuchando (solo micrófono)"
        elif started_sources == ["MEETING"]:
            status = "Escuchando (solo audio de reunión)"
        else:
            status = f"Escuchando · {self.engine_label}"

        self._emit("on_status", status)

    def stop(self) -> None:
        if not self.running and not self.transcriber:
            return

        self.running = False

        if self.mic_streamer:
            self.mic_streamer.stop()
            self.mic_streamer = None

        if self.loopback_streamer:
            self.loopback_streamer.stop()
            self.loopback_streamer = None

        if self.transcriber:
            self.transcriber.stop()
            self.transcriber = None

        self._emit("on_status", "Detenido")

    def _emit(self, name: str, *args) -> None:
        callback = self.callbacks.get(name)
        if callback:
            callback(*args)
