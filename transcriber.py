"""Transcripción local con Faster-Whisper.

El modelo se carga de forma perezosa para que importar la aplicación no bloquee
la interfaz ni consuma memoria antes de iniciar la captura.
"""

import threading

from faster_whisper import WhisperModel

from config import (
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL_SIZE,
)

_model = None
_model_lock = threading.Lock()
_transcribe_lock = threading.Lock()

TECH_VOCAB_HINT = (
    "Java, JDK, JVM, Spring Boot, Spring Framework, Hibernate, Maven, Gradle, "
    "microservices, REST API, endpoint, DTO, JPA, JWT, OAuth, singleton, SOLID, "
    "dependency injection, threads, concurrency, parallelism, deadlock, race condition, "
    "ExecutorService, CompletableFuture, async, database, SQL, NoSQL, MongoDB, "
    "PostgreSQL, MySQL, Redis, Kafka, RabbitMQ, Azure, AWS, Google Cloud, Docker, "
    "Kubernetes, App Service, Azure Functions, CI/CD, pipeline, build, deploy, rollback, "
    "GitHub Actions, Jenkins, unit test, integration test, test coverage, Terraform, "
    "SonarQube, code review, pull request, architecture, DDD, event driven, cache, scalability."
)


def _get_model():
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                print(f"Cargando Faster-Whisper '{WHISPER_MODEL_SIZE}'...")
                _model = WhisperModel(
                    WHISPER_MODEL_SIZE,
                    device=WHISPER_DEVICE,
                    compute_type=WHISPER_COMPUTE_TYPE,
                )
                print("Faster-Whisper cargado.")

    return _model


def transcribe_audio(audio_np, language="en"):
    """Transcribe un segmento mono a 16 kHz y devuelve texto limpio."""
    if audio_np is None or len(audio_np) == 0:
        return ""

    with _transcribe_lock:
        segments, _ = _get_model().transcribe(
            audio_np,
            language=language,
            beam_size=3,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            initial_prompt=TECH_VOCAB_HINT,
        )
        return " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text and segment.text.strip()
        ).strip()
