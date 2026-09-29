"""Traducción asíncrona de subtítulos EN -> ES usando DeepSeek."""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEEPSEEK_ENDPOINT = "https://api.deepseek.com"
REQUEST_TIMEOUT = 4.0

_client = None
_client_lock = __import__("threading").Lock()

TRANSLATE_SYSTEM_PROMPT = (
    "Translate the following English meeting subtitle into natural Spanish. "
    "Return ONLY the Spanish translation. Keep technical product names, acronyms, "
    "class names, cloud services and code identifiers unchanged when appropriate."
)


def _get_client():
    global _client

    if _client is None:
        with _client_lock:
            if _client is None:
                api_key = os.environ.get("DEEPSEEK_API_KEY")
                if not api_key:
                    return None

                _client = OpenAI(
                    base_url=DEEPSEEK_ENDPOINT,
                    api_key=api_key,
                    timeout=REQUEST_TIMEOUT,
                    max_retries=0,
                )

    return _client


def translate_text(text, source="EN", target="ES"):
    if not text or not text.strip():
        return ""

    if source.upper() != "EN" or target.upper() != "ES":
        return ""

    client = _get_client()
    if client is None:
        print("[Translator] DEEPSEEK_API_KEY no configurada.")
        return ""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=180,
        )

        content = response.choices[0].message.content
        return content.strip() if content else ""
    except Exception as exc:
        print(f"[Translator] Error: {exc}")
        return ""
