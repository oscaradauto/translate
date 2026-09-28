import sounddevice as sd
import soundcard as sc
import numpy as np
import torch
import queue
import threading
import time

from RealtimeSTT import AudioToTextRecorder

from assistant import is_question, answer_question, warmup
from config import get_language_mode, get_assistant_enabled, native_model_lock, get_assistant_listen_mode
from transcriber import transcribe_audio

TARGET_SAMPLE_RATE = 16000
MIC_BLOCK_SIZE = 512
SILENCE_TIMEOUT = 0.5
MIN_SPEECH_DURATION = 0.3
VAD_THRESHOLD = 0.5
MIC_DEVICE_INDEX = 1  # Headset Microphone (Jabra EVOLVE)
LOOPBACK_BLOCK_FRAMES = 1536

# Parámetros del streaming en tiempo real para tu propia voz (RealtimeSTT)
REALTIME_MODEL = "small"
REALTIME_POST_SPEECH_SILENCE = 0.4

print("Cargando modelo VAD (silero-vad)...")
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
print("Modelo VAD cargado.")


def _resample_linear(block, orig_sr, target_sr):
    if orig_sr == target_sr or len(block) == 0:
        return block.astype(np.float32)
    duration = len(block) / orig_sr
    target_len = max(1, int(round(duration * target_sr)))
    x_old = np.linspace(0, duration, num=len(block), endpoint=False)
    x_new = np.linspace(0, duration, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, block).astype(np.float32)


class RealtimeMicStreamer:
    """
    Captura tu micrófono con streaming en tiempo real (RealtimeSTT).
    Entrega texto parcial mientras hablas (on_partial_text) y la frase ya
    cerrada/limpia cuando terminas (on_final_text), similar al comportamiento
    de Microsoft Teams. El idioma de reconocimiento coincide siempre con el
    modo de reunión seleccionado ("en" o "es").
    """

    def __init__(self, source_label, device_index, on_partial_text, on_final_text, language="en"):
        self.source_label = source_label
        self.device_index = device_index
        self.on_partial_text = on_partial_text
        self.on_final_text = on_final_text
        self.language = language
        self.running = False
        self.thread = None
        self.recorder = None

    def _on_partial(self, text):
        if text:
            self.on_partial_text(self.source_label, text)

    def start(self):
        self.running = True
        self.recorder = AudioToTextRecorder(
            model=REALTIME_MODEL,
            language=self.language,
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
            except Exception as e:
                if self.running:
                    print(f"[{self.source_label}] RealtimeSTT error: {e}")
                continue
            if sentence and sentence.strip():
                self.on_final_text(self.source_label, sentence.strip())

    def stop(self):
        self.running = False
        if self.recorder:
            try:
                self.recorder.stop()
            except Exception as e:
                print(f"[{self.source_label}] Error al detener grabación: {e}")
            try:
                self.recorder.shutdown()
            except Exception as e:
                print(f"[{self.source_label}] Error al cerrar RealtimeSTT: {e}")
        if self.thread:
            self.thread.join(timeout=2.0)


class LoopbackVADStreamer:
    """Captura el audio de salida del sistema (voces de compañeros) usando 'soundcard'."""

    def __init__(self, source_label, on_speech_segment, on_error=None):
        self.source_label = source_label
        self.on_speech_segment = on_speech_segment
        self.on_error = on_error
        self.audio_queue = queue.Queue(maxsize=400)
        self.running = False
        self.capture_thread = None
        self.processing_thread = None
        self.samplerate = 48000

    def start(self):
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.processing_thread.start()

    def stop(self):
        self.running = False

    def _capture_loop(self):
        try:
            speaker = sc.default_speaker()
            mic = sc.get_microphone(speaker.name, include_loopback=True)
            with mic.recorder(samplerate=self.samplerate) as recorder:
                while self.running:
                    data = recorder.record(numframes=LOOPBACK_BLOCK_FRAMES)
                    mono = data.mean(axis=1) if data.ndim > 1 else data
                    try:
                        self.audio_queue.put_nowait(mono.copy())
                    except queue.Full:
                        pass
        except Exception as e:
            self.running = False
            if self.on_error:
                self.on_error(e)

    def _process_loop(self):
        _run_vad_loop(self.audio_queue, self.samplerate, self.source_label,
                      self.on_speech_segment, lambda: self.running)


def _run_vad_loop(audio_queue, samplerate, source_label, on_speech_segment, is_running):
    speech_buffer = []
    silence_duration = 0.0
    is_speaking = False
    resample_carry = np.array([], dtype=np.float32)
    REQUIRED_SAMPLES = 512

    while is_running():
        try:
            block = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        block_16k = _resample_linear(block, samplerate, TARGET_SAMPLE_RATE)
        if len(block_16k) == 0:
            continue

        resample_carry = np.concatenate([resample_carry, block_16k])

        while len(resample_carry) >= REQUIRED_SAMPLES:
            chunk = resample_carry[:REQUIRED_SAMPLES]
            resample_carry = resample_carry[REQUIRED_SAMPLES:]

            block_duration = REQUIRED_SAMPLES / TARGET_SAMPLE_RATE
            audio_tensor = torch.from_numpy(chunk)
            try:
                with native_model_lock:
                    speech_prob = vad_model(audio_tensor, TARGET_SAMPLE_RATE).item()
            except Exception as e:
                print(f"[{source_label}] VAD error: {e}")
                continue

            if speech_prob >= VAD_THRESHOLD:
                is_speaking = True
                silence_duration = 0.0
                speech_buffer.append(chunk)
            else:
                if is_speaking:
                    silence_duration += block_duration
                    speech_buffer.append(chunk)

                    if silence_duration >= SILENCE_TIMEOUT:
                        full_audio = np.concatenate(speech_buffer)
                        duration = len(full_audio) / TARGET_SAMPLE_RATE

                        if duration >= MIN_SPEECH_DURATION and on_speech_segment:
                            on_speech_segment(full_audio, source_label)

                        speech_buffer = []
                        is_speaking = False
                        silence_duration = 0.0


class ListenerController:

    def __init__(self, callbacks: dict, mic_device=MIC_DEVICE_INDEX):
        self.callbacks = callbacks
        self.mic_device = mic_device
        self.mic_streamer = None
        self.loopback_streamer = None
        self._segment_counter = 0
        self._counter_lock = threading.Lock()
        self._assistant_busy = threading.Event()

    def _next_segment_id(self):
        with self._counter_lock:
            self._segment_counter += 1
            return self._segment_counter

    def _emit(self, name, *args):
        cb = self.callbacks.get(name)
        if cb:
            cb(*args)

    def _should_trigger_assistant(self, source_label):
        mode = get_assistant_listen_mode()
        if mode == "ambos":
            return True
        if mode == "compañeros":
            return source_label == "Compañeros"
        if mode == "yo":
            return source_label == "Tú"
        return False

    def _handle_question(self, source, text):
        if self._assistant_busy.is_set():
            print("[Assistant] Ya hay una pregunta en proceso, se ignora esta.")
            return
        self._assistant_busy.set()
        try:
            self._emit("on_question", source, text)
            self._emit("on_assistant_state", "Pensando...")

            answer = answer_question(text)

            if answer:
                self._emit("on_answer", answer)
            else:
                self._emit("on_answer_error",
                           "No se pudo obtener respuesta (servicio no disponible). Intenta de nuevo.")
        finally:
            self._assistant_busy.clear()

    def _handle_subtitle(self, source, text, segment_id):
        # Sin traducción: solo se emite el subtítulo en el idioma original (EN)
        self._emit("on_subtitle", source, text, segment_id)

    # ---------- Flujo LOOPBACK (Compañeros): VAD + Whisper por segmento ----------
    def _on_segment(self, audio_np, source_label):
        text = transcribe_audio(audio_np)
        if not text or len(text.strip()) < 3:
            return

        mode = get_language_mode()
        if mode == "en":
            segment_id = self._next_segment_id()
            threading.Thread(
                target=self._handle_subtitle, args=(source_label, text, segment_id), daemon=True
            ).start()

        if self._should_trigger_assistant(source_label) and get_assistant_enabled() and is_question(text):
            threading.Thread(target=self._handle_question, args=(source_label, text), daemon=True).start()

    # ---------- Flujo MIC (Tú): streaming en tiempo real (RealtimeSTT) ----------
    def _on_mic_partial(self, source_label, text):
        # El subtítulo parcial depende SOLO del idioma de reunión, sin importar el asistente
        if get_language_mode() == "en":
            self._emit("on_subtitle_partial", source_label, text)

    def _on_mic_final(self, source_label, text):
        if not text or len(text.strip()) < 3:
            return

        mode = get_language_mode()
        if mode == "en":
            segment_id = self._next_segment_id()
            threading.Thread(
                target=self._handle_subtitle, args=(source_label, text, segment_id), daemon=True
            ).start()

        if self._should_trigger_assistant(source_label) and get_assistant_enabled() and is_question(text):
            threading.Thread(target=self._handle_question, args=(source_label, text), daemon=True).start()

    def start(self):
        if get_assistant_enabled():
            self._emit("on_status", "Preparando asistente...")
            warmup()

        mic_language = get_language_mode()
        self.mic_streamer = RealtimeMicStreamer(
            "Tú", self.mic_device, self._on_mic_partial, self._on_mic_final, language=mic_language
        )
        self.mic_streamer.start()

        def _on_loopback_error(e):
            print(f"[Loopback] No se pudo iniciar captura de sistema: {e}")
            self._emit("on_status", "Escuchando (solo tu voz, sin loopback)...")

        self.loopback_streamer = LoopbackVADStreamer("Compañeros", self._on_segment, on_error=_on_loopback_error)
        self.loopback_streamer.start()
        self._emit("on_status", "Escuchando...")

    def restart_mic_language(self, new_language):
        """
        Reinicia únicamente el streamer del micrófono con el nuevo idioma,
        sin afectar el loopback de 'Compañeros'. Se usa cuando el usuario
        cambia el modo de reunión (EN/ES) mientras la app sigue escuchando.
        """
        if not self.mic_streamer:
            return

        self._emit("on_status", "Actualizando idioma...")
        try:
            self.mic_streamer.stop()
        except Exception as e:
            print(f"[Tú] Error al detener streamer previo: {e}")

        self.mic_streamer = RealtimeMicStreamer(
            "Tú", self.mic_device, self._on_mic_partial, self._on_mic_final, language=new_language
        )
        self.mic_streamer.start()
        self._emit("on_status", "Escuchando...")

    def stop(self):
        if self.mic_streamer:
            self.mic_streamer.stop()
        if self.loopback_streamer:
            self.loopback_streamer.stop()
        self._emit("on_status", "Detenido")