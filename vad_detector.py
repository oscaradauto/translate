"""Captura y transcripción de las dos fuentes de audio de Meeting Subtitles.

MICROPHONE  -> YOU
SYSTEM AUDIO -> COMPANION

No contiene lógica de agente ni detección de preguntas. La primera etapa de V2
está dedicada exclusivamente a subtítulos.
"""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundcard as sc
import torch
from RealtimeSTT import AudioToTextRecorder

from config import (
    MIC_DEVICE_INDEX,
    TRANSLATION_TIMEOUT,
    get_meeting_language,
)
from transcriber import transcribe_audio
from translator import translate_text

TARGET_SAMPLE_RATE = 16000
LOOPBACK_BLOCK_FRAMES = 1536
SILENCE_TIMEOUT = 0.5
MIN_SPEECH_DURATION = 0.3
VAD_THRESHOLD = 0.5
REALTIME_MODEL = "small"
REALTIME_POST_SPEECH_SILENCE = 0.4

_vad_model = None
_vad_lock = threading.Lock()


def _get_vad_model():
    global _vad_model

    if _vad_model is None:
        with _vad_lock:
            if _vad_model is None:
                print("Cargando modelo VAD (silero-vad)...")
                _vad_model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    trust_repo=True,
                )
                print("Modelo VAD cargado.")

    return _vad_model


def _resample_linear(block, orig_sr, target_sr):
    if orig_sr == target_sr or len(block) == 0:
        return block.astype(np.float32)

    duration = len(block) / orig_sr
    target_len = max(1, int(round(duration * target_sr)))
    x_old = np.linspace(0, duration, num=len(block), endpoint=False)
    x_new = np.linspace(0, duration, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, block).astype(np.float32)


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
            enable_realtime_transcription=True,
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
        """Detiene RealtimeSTT evitando cerrar el pipe mientras text() lo usa."""
        self.running = False

        recorder = self.recorder
        if recorder:
            # text() puede estar bloqueado esperando datos. abort() le indica
            # a RealtimeSTT que interrumpa el flujo antes de cerrar recursos.
            try:
                recorder.abort()
            except Exception:
                pass

            try:
                recorder.stop()
            except Exception:
                pass

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)

        # No llamamos shutdown() aquí. En la versión instalada de RealtimeSTT
        # se observó que shutdown() intenta cerrar un FasterWhisperEngine que
        # no expone close(), generando otro error durante el cierre.
        self.thread = None
        self.recorder = None


class LoopbackVADStreamer:
    """Audio del sistema/Teams/Zoom/Meet: se etiqueta como COMPANION."""

    def __init__(self, on_speech_segment, on_error=None):
        self.on_speech_segment = on_speech_segment
        self.on_error = on_error
        self.audio_queue = queue.Queue(maxsize=100)
        self.running = False
        self.capture_thread = None
        self.processing_thread = None
        self.samplerate = 48000

    def start(self):
        self.running = True

        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            name="system-audio-capture",
            daemon=True,
        )
        self.processing_thread = threading.Thread(
            target=self._process_loop,
            name="system-audio-vad",
            daemon=True,
        )

        self.capture_thread.start()
        self.processing_thread.start()

    def stop(self):
        self.running = False

        for thread in (self.capture_thread, self.processing_thread):
            if thread and thread.is_alive():
                thread.join(timeout=2.0)

        self.capture_thread = None
        self.processing_thread = None

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

                    try:
                        self.audio_queue.put_nowait(mono.astype(np.float32, copy=True))
                    except queue.Full:
                        # Si el procesamiento se atrasa, descartamos el bloque
                        # más reciente para no acumular latencia infinita.
                        pass

        except Exception as exc:
            self.running = False
            if self.on_error:
                self.on_error(exc)

    def _process_loop(self):
        _run_vad_loop(
            self.audio_queue,
            self.samplerate,
            self.on_speech_segment,
            lambda: self.running,
        )


def _run_vad_loop(audio_queue, samplerate, on_speech_segment, is_running):
    speech_buffer = []
    silence_duration = 0.0
    is_speaking = False
    resample_carry = np.array([], dtype=np.float32)
    required_samples = 512

    while is_running():
        try:
            block = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        block_16k = _resample_linear(
            block,
            samplerate,
            TARGET_SAMPLE_RATE,
        )

        if len(block_16k) == 0:
            continue

        resample_carry = np.concatenate([resample_carry, block_16k])

        while len(resample_carry) >= required_samples:
            chunk = resample_carry[:required_samples]
            resample_carry = resample_carry[required_samples:]

            try:
                speech_prob = _get_vad_model()(
                    torch.from_numpy(chunk),
                    TARGET_SAMPLE_RATE,
                ).item()
            except Exception as exc:
                print(f"[COMPANION] VAD error: {exc}")
                continue

            block_duration = required_samples / TARGET_SAMPLE_RATE

            if speech_prob >= VAD_THRESHOLD:
                is_speaking = True
                silence_duration = 0.0
                speech_buffer.append(chunk)
                continue

            if not is_speaking:
                continue

            silence_duration += block_duration
            speech_buffer.append(chunk)

            if silence_duration < SILENCE_TIMEOUT:
                continue

            full_audio = np.concatenate(speech_buffer)
            duration = len(full_audio) / TARGET_SAMPLE_RATE

            if duration >= MIN_SPEECH_DURATION:
                on_speech_segment(full_audio, "COMPANION")

            speech_buffer = []
            is_speaking = False
            silence_duration = 0.0


class ListenerController:
    """Orquesta exclusivamente subtítulos y traducción."""

    def __init__(self, callbacks, mic_device=MIC_DEVICE_INDEX):
        self.callbacks = callbacks
        self.mic_device = mic_device
        self.mic_streamer = None
        self.loopback_streamer = None
        self.running = False

        self._segment_counter = 0
        self._counter_lock = threading.Lock()

        # Máximo dos traducciones simultáneas: evita crear un hilo ilimitado
        # por cada frase de la reunión.
        self._translation_pool = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="subtitle-translation",
        )

    def _next_segment_id(self):
        with self._counter_lock:
            self._segment_counter += 1
            return self._segment_counter

    def _emit(self, name, *args):
        callback = self.callbacks.get(name)
        if callback:
            callback(*args)

    def _handle_subtitle(self, source, text):
        segment_id = self._next_segment_id()

        # EN aparece inmediatamente.
        self._emit("on_subtitle", source, text, segment_id)

        # ES se procesa aparte para no bloquear la aparición del EN.
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

            if translated:
                self._emit(
                    "on_subtitle_translated",
                    source,
                    translated,
                    segment_id,
                )

        future.add_done_callback(_translation_done)

    def _on_companion_segment(self, audio_np, source_label):
        if not self.running:
            return

        text = transcribe_audio(
            audio_np,
            language=get_meeting_language(),
        )

        if len(text.strip()) < 2:
            return

        self._handle_subtitle(source_label, text)

    def _on_mic_partial(self, source_label, text):
        if self.running and text:
            self._emit("on_subtitle_partial", source_label, text)

    def _on_mic_final(self, source_label, text):
        if not self.running or len(text.strip()) < 2:
            return

        self._handle_subtitle(source_label, text)

    def start(self):
        if self.running:
            return

        self.running = True
        self._emit("on_status", "Preparando audio...")

        try:
            self.mic_streamer = RealtimeMicStreamer(
                self.mic_device,
                self._on_mic_partial,
                self._on_mic_final,
            )
            self.mic_streamer.start()

            self.loopback_streamer = LoopbackVADStreamer(
                self._on_companion_segment,
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
