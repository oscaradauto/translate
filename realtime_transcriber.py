"""Low-latency OpenAI realtime transcription client for Stage 1.

This module owns only the WebSocket connection and transcription events.
Audio capture is handled by the source-specific classes in vad_detector.py.
"""

from __future__ import annotations

import base64
import json
import os
import threading
from typing import Callable

import websocket

OPENAI_REALTIME_URL = (
    "wss://api.openai.com/v1/realtime?model=gpt-live-transcribe"
)

PartialCallback = Callable[[str, str, str], None]
FinalCallback = Callable[[str, str, str], None]
StatusCallback = Callable[[str, str], None]
ErrorCallback = Callable[[str, Exception], None]


class RealtimeTranscriptionSession:
    """One persistent gpt-live-transcribe session for one audio source."""

    def __init__(
        self,
        source: str,
        on_partial: PartialCallback,
        on_final: FinalCallback,
        on_status: StatusCallback | None = None,
        on_error: ErrorCallback | None = None,
        model: str = "gpt-live-transcribe",
        delay: str = "low",
        prompt: str = "",
        keywords: list[str] | None = None,
    ) -> None:
        self.source = source
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_status = on_status
        self.on_error = on_error
        self.model = model
        self.delay = delay
        self.prompt = prompt
        self.keywords = keywords or []

        self._running = False
        self._closing = False
        self._closed_event = threading.Event()
        self._ready_event = threading.Event()
        self._send_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._ws: websocket.WebSocket | None = None
        self._audio_buffered = False
        self._partial_buffers: dict[str, str] = {}

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, timeout: float = 12.0) -> None:
        if self._running:
            return

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY no está configurada. "
                "Agrega tu API key en el archivo .env."
            )

        self._running = True
        self._closing = False
        self._closed_event.clear()
        self._ready_event.clear()
        self._partial_buffers.clear()
        self._audio_buffered = False

        self._thread = threading.Thread(
            target=self._run,
            name=f"openai-transcription-{self.source.lower()}",
            daemon=True,
        )
        self._thread.start()

        if not self._ready_event.wait(timeout):
            self._running = False
            self._close_socket()
            raise TimeoutError(
                f"No se pudo establecer la sesión de transcripción para "
                f"{self.source}."
            )

    def append_audio(self, pcm16: bytes) -> bool:
        if not pcm16 or not self._running or self._closing:
            return False

        event = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm16).decode("ascii"),
        }

        try:
            self._send(event)
            self._audio_buffered = True
            return True
        except Exception as exc:
            self._report_error(exc)
            return False

    def commit(self) -> bool:
        if not self._running or self._closing or not self._audio_buffered:
            return False

        try:
            self._send({"type": "input_audio_buffer.commit"})
            self._audio_buffered = False
            return True
        except Exception as exc:
            self._report_error(exc)
            return False

    def stop(self) -> None:
        """Close the session gracefully and wait for final events."""
        thread = self._thread
        ws = self._ws

        if not self._running and not thread:
            return

        self._running = False

        if ws:
            self._closing = True

            # Flush the current audio turn before closing the transport.
            if self._audio_buffered:
                try:
                    self._send({"type": "input_audio_buffer.commit"})
                except Exception:
                    pass
                self._audio_buffered = False

            try:
                self._send({"type": "session.close"})
            except Exception:
                pass

            # Give the receive loop time to consume final transcription events.
            self._closed_event.wait(5.0)

        self._close_socket()

        if thread and thread.is_alive():
            thread.join(timeout=2.0)

        self._thread = None
        self._ws = None
        self._closing = False
        self._ready_event.clear()

    def _run(self) -> None:
        try:
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            ws = websocket.create_connection(
                OPENAI_REALTIME_URL,
                header=[f"Authorization: Bearer {api_key}"],
                timeout=5.0,
                ping_interval=20,
                ping_timeout=10,
            )
            ws.settimeout(1.0)
            self._ws = ws

            self._notify_status("connecting")

            # Wait for the server to create the session before configuring it.
            while self._running and not self._closing:
                event = self._recv_event()
                if event is None:
                    continue
                if event.get("type") == "session.created":
                    break
                if event.get("type") == "error":
                    raise RuntimeError(self._format_error(event))

            if not self._running:
                return

            self._send(
                {
                    "type": "session.update",
                    "session": {
                        "type": "transcription",
                        "audio": {
                            "input": {
                                "format": {
                                    "type": "audio/pcm",
                                    "rate": 24000,
                                },
                                "transcription": {
                                    "model": self.model,
                                    "prompt": self.prompt,
                                    "keywords": self.keywords,
                                    "languages": ["en"],
                                    "delay": self.delay,
                                },
                                "turn_detection": None,
                            }
                        },
                    },
                }
            )

            while self._running and not self._closing:
                event = self._recv_event()
                if event is None:
                    continue

                event_type = event.get("type")

                if event_type == "session.updated":
                    self._ready_event.set()
                    self._notify_status("connected")
                    continue

                if event_type == "conversation.item.input_audio_transcription.delta":
                    item_id = str(event.get("item_id", ""))
                    delta = event.get("delta", "")
                    if item_id and delta:
                        text = self._partial_buffers.get(item_id, "") + delta
                        self._partial_buffers[item_id] = text
                        self.on_partial(self.source, text, item_id)
                    continue

                if (
                    event_type
                    == "conversation.item.input_audio_transcription.completed"
                ):
                    item_id = str(event.get("item_id", ""))
                    transcript = event.get("transcript", "").strip()
                    self._partial_buffers.pop(item_id, None)
                    if transcript:
                        self.on_final(self.source, transcript, item_id)
                    continue

                if event_type == "session.closed":
                    self._closed_event.set()
                    break

                if event_type == "error":
                    raise RuntimeError(self._format_error(event))

        except Exception as exc:
            self._report_error(exc)
        finally:
            self._closed_event.set()
            self._close_socket()

    def _recv_event(self) -> dict | None:
        ws = self._ws
        if ws is None:
            return None

        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            return None
        except (
            websocket.WebSocketConnectionClosedException,
            websocket.WebSocketBadStatusException,
        ) as exc:
            if self._running and not self._closing:
                self._report_error(exc)
            return None

        if not raw:
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _send(self, event: dict) -> None:
        ws = self._ws
        if ws is None:
            raise RuntimeError("WebSocket de transcripción no disponible.")

        payload = json.dumps(event, ensure_ascii=False)
        with self._send_lock:
            ws.send(payload)

    def _close_socket(self) -> None:
        ws = self._ws
        if ws is None:
            return

        with self._send_lock:
            try:
                ws.close()
            except Exception:
                pass

    def _notify_status(self, status: str) -> None:
        if self.on_status:
            self.on_status(self.source, status)

    def _report_error(self, exc: Exception) -> None:
        if self.on_error:
            self.on_error(self.source, exc)
