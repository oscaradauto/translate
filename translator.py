"""Traducción rápida y local de subtítulos EN -> ES usando Ollama."""

import json
import os
import threading
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()

OLLAMA_ENDPOINT = os.environ.get(
    "OLLAMA_ENDPOINT",
    "http://localhost:11434/api/generate",
)

# Gemma 2B está orientado a lenguaje natural, por lo que es una mejor base
# para traducción que un modelo especializado en código.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma2:2b")

# Mantener el modelo cargado evita el coste de volver a cargarlo entre frases.
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "10m")
REQUEST_TIMEOUT = float(os.environ.get("TRANSLATION_TIMEOUT_SECONDS", "8.0"))

_client_lock = threading.Lock()

TRANSLATE_PROMPT = """You are a real-time meeting subtitle translator.

Translate the English subtitle into natural, neutral Spanish.

Rules:
- Return ONLY the Spanish translation.
- Do not explain, summarize, or add information.
- Preserve the original meaning and intent.
- Prefer natural spoken Spanish over literal word-for-word translation.
- Keep the translation concise because it is displayed as a live subtitle.
- Keep technical product names, company names, acronyms, class names,
  method names, API names, technologies, cloud services, and code identifiers
  unchanged when appropriate.
- Preserve names of people.
- Keep common technical terms in English when that is how they are normally
  used by software teams (for example: pull request, deploy, endpoint, commit,
  rollback, pipeline, build, branch, merge, framework).
"""

# Gemma puede generar una respuesta limpia, pero limitamos su salida para
# evitar explicaciones accidentales y mantener baja la latencia.
OLLAMA_OPTIONS = {
    "temperature": 0.0,
    "num_predict": 128,
}


def translate_text(text, source="EN", target="ES"):
    """Translate an English subtitle to Spanish using the local Ollama model."""
    if not text or not text.strip():
        return ""

    if source.upper() != "EN" or target.upper() != "ES":
        return ""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{TRANSLATE_PROMPT}\n{text.strip()}",
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": OLLAMA_OPTIONS,
    }

    request = urllib.request.Request(
        OLLAMA_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        # Ollama local no necesita múltiples requests concurrentes para una
        # misma traducción; serializarlos evita cargar innecesariamente la CPU.
        with _client_lock:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                body = json.loads(response.read().decode("utf-8"))

        content = body.get("response", "")
        return content.strip() if content else ""
    except urllib.error.URLError as exc:
        print(f"[Translator] Ollama no disponible: {exc}")
        return ""
    except Exception as exc:
        print(f"[Translator] Error: {exc}")
        return ""
