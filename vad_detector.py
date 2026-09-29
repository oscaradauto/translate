"""Captura y transcripción de las dos fuentes de audio de Meeting Subtitles.

MICROPHONE  -> YOU
SYSTEM AUDIO -> COMPANION

No contiene lógica de agente ni detección de preguntas. La primera etapa de V2
está dedicada exclusivamente a subtítulos y traducción incremental.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundcard as sc
from RealtimeSTT import AudioToTextRecorder

from config import (
    MIC_DEVICE_INDEX,
    REALTIME_COMPUTE_TYPE,
    REALTIME_DEVICE,
    TRANSLATION_TIMEOUT,
    get_meeting_language,
)
from translator import translate_text

TARGET_SAMPLE_RATE = 16000
LOOPBACK_BLOCK_FRAMES = 1536
REALTIME_MODEL = "small"
REALTIME_POST_SPEECH_SILENCE = 0.4
REALTIME_PROCESSING_PAUSE = 0.2

# Traducción incremental: actualizamos el español aproximadamente cada segundo
# mientras el inglés continúa creciendo, sin enviar una petición por cada token.
INCREMENTAL_TRANSLATION_INTERVAL = 0.9
MIN_TRANSLATION_CHANGE_CHARS = 4


def _resample_linear(block, orig_sr, target_sr):
    if orig_sr == target_sr or len(block) == 0:
        return block.astype(np.float32)

    duration = len(block) / orig_sr
    target_len = max(1, int(round(duration * target_sr)))
    x_old = np.linspace(0, duration, num=len(block), endpoint=False)
    x_new = np.linspace(0, duration, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, block).astype(np.float32)


def _to_pcm16(audio_np):
    clipped = np.clip(audio_np, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


class RealtimeMicStreamer:
    """Micrófono físico: la voz del usuario siempre se etiqueta como YOU."""

    def __init__(self, device_index, on_partial_text, on_final_text):
        self.device_index = device_index
        self.on_partial_text = on_partial_text
        self.on_final_text = on_final_text
        self.running = False
        self.thread = None
        self.recorder = None

    def _on_partial(self, text):
        if text:
            self.on_partial_text("YOU", text)

    def start(self):
        self.running = True
        self.recorder = AudioToTextRecorder(
            model=REALTIME_MODEL,
            language=get_meeting_language(),
            spinner=False,
            compute_type=REALTIME_COMPUTE_TYPE,
            device=REALTIME_DEVICE,
            enable_realtime_transcription=True,
            realtime_processing_pause=REALTIME_PROCESSING_PAUSE,
            on_realtime_transcription_update=self._on_partial,
            use_microphone=True,
            input_device_index=self.device_index,
            post_speech_silence_duration=REALTIME_POST_SPEECH_SILENCE,
        )
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            try:
                sentence = self.recorder.text()
            except Exception as exc:
                if self.running:
                    print(f"[YOU] RealtimeSTT error: {exc}")
                continue

            if sentence and sentence.strip():
                self.on_final_text("YOU", sentence.strip())

    def stop(self):
        """Detiene RealtimeSTT y libera sus recursos de forma segura."""
        self.running = False

        recorder = self.recorder
        if recorder:
            try:
                recorder.abort()
            except Exception:
                pass

            try:
                recorder.realtime_transcription_model = None
            except Exception:
                pass

            try:
                recorder.shutdown()
            except Exception as exc:
                print(f"[YOU] RealtimeSTT shutdown error: {exc}")

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)

        self.thread = None
        self.recorder = None


class RealtimeSystemAudioStreamer:
    """Audio del sistema/Teams/Zoom/Meet con subtítulos parciales en tiempo real."""

    def __init__(self, on_partial_text, on_final_text, on_error=None):
        self.on_partial_text = on_partial_text
        self.on_final_text = on_final_text
        self.on_error = on_error
        self.running = False
        self.capture_thread = None
        self.transcription_thread = None
        self.recorder = None
        self.samplerate = 48000

    def _on_partial(self, text):
        if text:
            self.on_partial_text("COMPANION", text)

    def start(self):
        self.running = True

        try:
            self.recorder = AudioToTextRecorder(
                model=REALTIME_MODEL,
                language=get_meeting_language(),
                spinner=False,
                compute_type=REALTIME_COMPUTE_TYPE,
                device=REALTIME_DEVICE,
                enable_realtime_transcription=True,
                realtime_processing_pause=REALTIME_PROCESSING_PAUSE,
                on_realtime_transcription_update=self._on_partial,
                use_microphone=False,
                post_speech_silence_duration=REALTIME_POST_SPEECH_SILENCE,
            )

            self.capture_thread = threading.Thread(
                target=self._capture_loop,
                name="system-audio-capture",
                daemon=True,
            )
            self.transcription_thread = threading.Thread(
                target=self._transcription_loop,
                name="system-audio-transcription",
                daemon=True,
            )

            self.capture_thread.start()
            self.transcription_thread.start()
        except Exception as exc:
            self.running = False
            self._safe_shutdown_recorder()
            if self.on_error:
                self.on_error(exc)

    def _capture_loop(self):
        try:
            speaker = sc.default_speaker()
            loopback_mic = sc.get_microphone(
                speaker.name,
                include_loopback=True,
            )

            with loopback_mic.recorder(samplerate=self.samplerate) as recorder:
                while self.running:
                    data = recorder.record(numframes=LOOPBACK_BLOCK_FRAMES)
                    mono = data.mean(axis=1) if data.ndim > 1 else data

                    audio_16k = _resample_linear(
                        mono,
                        self.samplerate,
                        TARGET_SAMPLE_RATE,
                    )
                    if len(audio_16k):
                        self.recorder.feed_audio(
                            _to_pcm16(audio_16k),
                            original_sample_rate=TARGET_SAMPLE_RATE,
                        )
        except Exception as exc:
            if self.running:
                self.running = False
                if self.on_error:
                    self.on_error(exc)

    def _transcription_loop(self):
        while self.running:
            try:
                sentence = self.recorder.text()
            except Exception as exc:
                if self.running:
                    print(f"[COMPANION] RealtimeSTT error: {exc}")
                continue

            if sentence and sentence.strip():
                self.on_final_text("COMPANION", sentence.strip())

    def _safe_shutdown_recorder(self):
        recorder = self.recorder
        if not recorder:
            return

        try:
            recorder.abort()
        except Exception:
            pass

        try:
            recorder.realtime_transcription_model = None
        except Exception:
            pass

        try:
            recorder.shutdown()
        except Exception as exc:
            print(f"[COMPANION] RealtimeSTT shutdown error: {exc}")

        self.recorder = None

    def stop(self):
        self.running = False

        recorder = self.recorder
        if recorder:
            try:
                recorder.abort()
            except Exception:
                pass

        for thread in (self.capture_thread, self.transcription_thread):
            if thread and thread.is_alive():
                thread.join(timeout=3.0)

        self._safe_shutdown_recorder()

        self.capture_thread = None
        self.transcription_thread = None


class ListenerController:
    """Orquesta subtítulos, traducción incremental y traducción final."""

    def __init__(self, callbacks, mic_device=MIC_DEVICE_INDEX):
        self.callbacks = callbacks
        self.mic_device = mic_device
        self.mic_streamer = None
        self.loopback_streamer = None
        self.running = False

        self._segment_counter = 0
        self._counter_lock = threading.Lock()

        self._translation_pool = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="subtitle-translation",
        )

        # Estado independiente por fuente. La generación evita que una
        # traducción antigua pueda sobrescribir una más reciente.
        self._translation_lock = threading.Lock()
        self._translation_state = {
            "YOU": {
                "last_submitted_text": "",
                "last_submit_time": 0.0,
                "generation": 0,
            },
            "COMPANION": {
                "last_submitted_text": "",
                "last_submit_time": 0.0,
                "generation": 0,
            },
        }

    def _next_segment_id(self):
        with self._counter_lock:
            self._segment_counter += 1
            return self._segment_counter

    def _emit(self, name, *args):
        callback = self.callbacks.get(name)
        if callback:
            callback(*args)

    def _submit_translation(self, source, text, segment_id, is_final=False):
        text = text.strip()
        if not text:
            return

        with self._translation_lock:
            state = self._translation_state[source]
            state["generation"] += 1
            generation = state["generation"]
            state["last_submitted_text"] = text
            state["last_submit_time"] = time.monotonic()

        future = self._translation_pool.submit(
            translate_text,
            text,
            "EN",
            "ES",
        )

        def _translation_done(done):
            try:
                translated = done.result(timeout=TRANSLATION_TIMEOUT)
            except Exception as exc:
                print(f"[Translation] Error: {exc}")
                return

            if not translated or not self.running:
                return

            # Si llegó una traducción más nueva, ignoramos esta respuesta.
            with self._translation_lock:
                current_generation = self._translation_state[source]["generation"]

            if generation != current_generation:
                return

            self._emit(
                "on_subtitle_translated",
                source,
                translated,
                segment_id,
                not is_final,
            )

        future.add_done_callback(_translation_done)

    def _on_partial(self, source_label, text):
        if not self.running or not text:
            return

        self._emit("on_subtitle_partial", source_label, text)

        # No traducimos cada actualización de RealtimeSTT. Esperamos ~0.9 s
        # o un cambio suficientemente grande para mantener baja la latencia.
        now = time.monotonic()
        text = text.strip()

        with self._translation_lock:
            state = self._translation_state[source_label]
            elapsed = now - state["last_submit_time"]
            changed = abs(
                len(text) - len(state["last_submitted_text"])
            ) >= MIN_TRANSLATION_CHANGE_CHARS

        if (
            elapsed >= INCREMENTAL_TRANSLATION_INTERVAL
            and changed
            and len(text) >= 4
        ):
            self._submit_translation(
                source_label,
                text,
                segment_id=0,
                is_final=False,
            )

    def _on_final(self, source_label, text):
        if not self.running or len(text.strip()) < 2:
            return

        segment_id = self._next_segment_id()
        self._emit("on_subtitle", source_label, text.strip(), segment_id)

        # La traducción final siempre se envía, aunque haya habido traducciones
        # incrementales anteriores.
        self._submit_translation(
            source_label,
            text,
            segment_id=segment_id,
            is_final=True,
        )

    def start(self):
        if self.running:
            return

        self.running = True

        with self._translation_lock:
            for state in self._translation_state.values():
                state["last_submitted_text"] = ""
                state["last_submit_time"] = 0.0
                state["generation"] = 0

        self._emit("on_status", "Preparando audio...")

        try:
            self.mic_streamer = RealtimeMicStreamer(
                self.mic_device,
                self._on_partial,
                self._on_final,
            )
            self.mic_streamer.start()

            self.loopback_streamer = RealtimeSystemAudioStreamer(
                self._on_partial,
                self._on_final,
                on_error=self._on_loopback_error,
            )
            self.loopback_streamer.start()

            self._emit("on_status", "Escuchando...")
        except Exception as exc:
            print(f"[Audio] Error iniciando captura: {exc}")
            self.stop()
            self._emit("on_status", "Error de audio")

    def _on_loopback_error(self, exc):
        print(f"[COMPANION] No se pudo capturar audio del sistema: {exc}")
        self._emit("on_status", "Escuchando (solo YOU)")

    def stop(self):
        if not self.running and not self.mic_streamer and not self.loopback_streamer:
            return

        self.running = False

        if self.mic_streamer:
            self.mic_streamer.stop()

        if self.loopback_streamer:
            self.loopback_streamer.stop()

        self.mic_streamer = None
        self.loopback_streamer = None

        self._translation_pool.shutdown(wait=False, cancel_futures=True)
        self._translation_pool = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="subtitle-translation",
        )

        self._emit("on_status", "Detenido")
