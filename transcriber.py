from faster_whisper import WhisperModel
from config import get_language_mode

print("Cargando modelo Whisper...")
MODEL_SIZE = "small"
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print(f"Modelo Whisper '{MODEL_SIZE}' cargado.")

TECH_VOCAB_HINT = (
    # Java / Backend
    "Java, Java, programación en Java, lenguaje Java, " 
    "Java, JDK, JVM, Spring Boot, Spring Framework, Hibernate, Maven, Gradle, "
    "microservicios, microservices, API REST, endpoint, DTO, JPA, JWT, OAuth, "
    "singleton, patrón de diseño, design pattern, SOLID, DRY, KISS, "
    "inyección de dependencias, dependency injection, "
    # Concurrencia / Hilos
    "hilos, threads, concurrencia, concurrency, paralelismo, parallelism, "
    "deadlock, race condition, mutex, semáforo, semaphore, thread pool, "
    "ExecutorService, CompletableFuture, async, asincrono, sincrono, "
    # Bases de datos
    "base de datos, database, SQL, NoSQL, MongoDB, PostgreSQL, MySQL, Redis, "
    "índice, index, query, transacción, transaction, ACID, sharding, "
    # Cloud / Azure
    "Azure, AWS, Google Cloud, contenedor, container, Docker, Kubernetes, "
    "App Service, Azure Functions, Blob Storage, "
    # CI/CD
    "CI/CD, continuous integration, continuous delivery, continuous deployment, "
    "integración continua, entrega continua, despliegue continuo, "
    "pipeline, build, deploy, rollback, GitHub Actions, GitLab CI, Jenkins, "
    "Azure DevOps, Azure Pipelines, Bitbucket Pipelines, Travis CI, CircleCI, "
    "artifact, artefacto, runner, staging, producción, production, "
    "quality gate, code review, pull request, merge request, "
    "unit test, integration test, test coverage, "
    "Terraform, infraestructura como código, infrastructure as code, "
    "SonarQube, linting, versionado semántico, semantic versioning, "
    # Arquitectura general
    "arquitectura hexagonal, clean architecture, DDD, event driven, "
    "message broker, Kafka, RabbitMQ, cache, caching, escalabilidad, scalability."
)


def transcribe_audio(audio_np):
    """
    Transcribe un array de audio (float32, 16kHz, mono).
    Usa el idioma seleccionado en la UI (config.LANGUAGE_MODE).
    """
    lang = get_language_mode()  # "en" o "es"
    segments, info = model.transcribe(
        audio_np,
        language=lang,
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        initial_prompt=TECH_VOCAB_HINT,
    )
    text = " ".join([seg.text.strip() for seg in segments]).strip()
    return text