"""Groq Stage 2 preflight checks and user-friendly API errors."""

from __future__ import annotations

import io
import os
import wave
from dataclasses import dataclass

from groq import Groq

from config import GROQ_MODEL, GROQ_STT_MODEL


@dataclass(frozen=True)
class GroqPreflightResult:
    ready: bool
    gpt_ok: bool
    whisper_ok: bool
    message: str


def _status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _header(exc: Exception, name: str) -> str:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return ""

    try:
        return str(headers.get(name, "") or "").strip()
    except Exception:
        return ""


def describe_groq_error(
    exc: Exception,
    service: str,
) -> str:
    """Return a concise error suitable for the Stage 2 UI."""
    status = _status_code(exc)

    if status == 429:
        hints = []
        retry_after = _header(exc, "retry-after")
        token_reset = _header(exc, "x-ratelimit-reset-tokens")
        request_reset = _header(exc, "x-ratelimit-reset-requests")

        if retry_after:
            hints.append(f"retry-after: {retry_after}")
        if token_reset:
            hints.append(f"token reset: {token_reset}")
        if request_reset:
            hints.append(f"request reset: {request_reset}")

        suffix = f" ({'; '.join(hints)})" if hints else ""
        return (
            f"{service}: límite de Groq alcanzado (HTTP 429){suffix}. "
            "La entrevista puede continuar escuchando, pero este servicio "
            "no podrá responder hasta que el límite se restablezca."
        )

    if status == 401:
        return (
            f"{service}: API key de Groq inválida o no autorizada "
            "(HTTP 401). Revisa GROQ_API_KEY."
        )

    if status == 403:
        return (
            f"{service}: acceso denegado por Groq (HTTP 403). "
            "Revisa el proyecto, el plan y los permisos del modelo."
        )

    if status is not None and status >= 500:
        return (
            f"{service}: Groq está temporalmente no disponible "
            f"(HTTP {status}). Intenta nuevamente en unos momentos."
        )

    detail = " ".join(str(exc).strip().split())
    if len(detail) > 240:
        detail = detail[:237] + "..."

    return (
        f"{service}: {detail}"
        if detail
        else f"{service}: error desconocido de Groq."
    )


def _silent_wav(seconds: float = 1.0) -> bytes:
    frames = int(16000 * seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * frames)
    return buffer.getvalue()


def verify_groq_services() -> GroqPreflightResult:
    """Make minimal real requests to both Groq services used by Stage 2."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return GroqPreflightResult(
            ready=False,
            gpt_ok=False,
            whisper_ok=False,
            message=(
                "✗ API key\n"
                "GROQ_API_KEY no está configurada en el archivo .env."
            ),
        )

    client = Groq(api_key=api_key)
    lines = ["✓ API key configurada"]
    whisper_ok = False
    gpt_ok = False

    try:
        client.audio.transcriptions.create(
            file=("groq-preflight.wav", _silent_wav()),
            model=GROQ_STT_MODEL,
            response_format="json",
            temperature=0.0,
        )
        whisper_ok = True
        lines.append(f"✓ Whisper: {GROQ_STT_MODEL}")
    except Exception as exc:
        lines.append(
            "✗ " + describe_groq_error(
                exc,
                f"Whisper {GROQ_STT_MODEL}",
            )
        )

    try:
        client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly OK.",
                }
            ],
            reasoning_effort="low",
            include_reasoning=False,
            max_completion_tokens=32,
            temperature=0.0,
        )
        gpt_ok = True
        lines.append(f"✓ GPT: {GROQ_MODEL}")
    except Exception as exc:
        lines.append(
            "✗ " + describe_groq_error(
                exc,
                f"GPT {GROQ_MODEL}",
            )
        )

    ready = whisper_ok and gpt_ok
    lines.append(
        ""
        + (
            "✓ Stage 2 listo para entrevista."
            if ready
            else "⚠ Stage 2 NO está listo para una entrevista."
        )
    )

    return GroqPreflightResult(
        ready=ready,
        gpt_ok=gpt_ok,
        whisper_ok=whisper_ok,
        message="\n".join(lines),
    )
