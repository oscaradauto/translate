"""Traducción asíncrona de subtítulos EN -> ES usando Ollama local."""

import json
import os
import threading
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()

OLLAMA_ENDPOINT = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:1.5b")
REQUEST_TIMEOUT = 10.0

_client_lock = threading.Lock()

TRANSLATE_SYSTEM_PROMPT = (
    "Translate the following English meeting subtitle into natural Spanish. "
    "Return ONLY the Spanish translation. Keep technical product names, acronyms, "
    "class names, cloud services and code identifiers unchanged when appropriate."
)


def translate_text(text, source="EN", target="ES"):
    """Translate an English subtitle to Spanish using the local Ollama model."""
    if not text or not text.strip():
        return ""

    if source.upper() != "EN" or target.upper() != "ES":
        return ""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{TRANSLATE_SYSTEM_PROMPT}\n\n{text.strip()}",
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.0,
        },
    }

    request = urllib.request.Request(
        OLLAMA_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
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
