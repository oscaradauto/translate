"""Desktop UI with two fully independent meeting modes.

Subtítulo:
    English-only live captions for a ~6-person meeting using local
    Faster-Whisper. No assistant logic is loaded or displayed.

Asistente:
    Independent technical-interview workspace for Stage 2. It uses Groq
    Whisper for high-accuracy interview transcription and Groq for answers.
"""

from __future__ import annotations

import sys
import threading
import time

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


# Keep the visual identity of the previous app.
APP_BG = "#0f1216"
CARD_BG = "#151a20"
CARD_BG_SOFT = "#181e25"
BORDER = "#2a323c"
TEXT = "#f0f2f5"
MUTED = "#8f9baa"
MUTED_DARK = "#687584"
BLUE = "#4da6ff"
GREEN = "#5cb85c"
ORANGE = "#ffa64d"
WARNING = "#f0ad4e"
RED = "#d9534f"


def _now() -> str:
    return time.strftime("%I:%M:%S %p")


def _button_style(primary: bool = False) -> str:
    if primary:
        return f"""
        QPushButton {{
            background-color: {BLUE};
            color: white;
            border-radius: 16px;
            padding: 7px 16px;
            border: none;
            font-weight: 700;
        }}
        QPushButton:hover {{
            background-color: #69b4ff;
        }}
        QPushButton:disabled {{
            background-color: #26313c;
            color: #6f7b88;
        }}
        """

    return """
    QPushButton {
        background-color: rgba(255,255,255,15);
        color: #f0f2f5;
        border-radius: 16px;
        padding: 7px 14px;
        border: 1px solid rgba(255,255,255,25);
        font-weight: 600;
    }
    QPushButton:hover {
        background-color: rgba(255,255,255,25);
    }
    QPushButton:disabled {
        color: #6f7b88;
        background-color: #1b2026;
        border-color: #2b323a;
    }
    """


class SubtitleBridge(QObject):
    status_changed = pyqtSignal(str)
    subtitle_partial = pyqtSignal(str, str, int)
    subtitle_ready = pyqtSignal(str, str, int)


class AssistantBridge(QObject):
    status_changed = pyqtSignal(str)
    transcript_partial = pyqtSignal(str, str, int)
    transcript_final = pyqtSignal(str, str, int)
    transcript_rejected = pyqtSignal(str, str)
    question_candidate = pyqtSignal(str)
    turn_understood = pyqtSignal(str, str, str)
    question_waiting = pyqtSignal(str)
    question_detected = pyqtSignal(str)
    question_ignored = pyqtSignal(str)
    answer_started = pyqtSignal(str)
    answer_delta = pyqtSignal(str)
    answer_completed = pyqtSignal(str)
    assistant_error = pyqtSignal(str)


class CaptionCard(QFrame):
    """A single live caption block updated by partial/final events."""

    def __init__(self, source: str, timestamp: str):
        super().__init__()
        self.source = source
        self.setObjectName("captionCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(8)

        source_name = "YOU" if source == "YOU" else "MEETING"
        source_color = BLUE if source == "YOU" else ORANGE

        badge = QLabel(source_name)
        badge.setStyleSheet(
            f"background-color: {source_color}; color: white; "
            "font-size: 9px; font-weight: 800; border-radius: 8px; "
            "padding: 3px 8px;"
        )

        self.time_label = QLabel(timestamp)
        self.time_label.setStyleSheet(
            f"color: {MUTED_DARK}; font-size: 10px; background: transparent;"
        )

        self.state_label = QLabel("")
        self.state_label.setStyleSheet(
            f"color: {MUTED_DARK}; font-size: 9px; background: transparent;"
        )

        header.addWidget(badge)
        header.addWidget(self.state_label)
        header.addStretch()
        header.addWidget(self.time_label)

        self.text_label = QLabel("")
        self.text_label.setWordWrap(True)
        self.text_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.text_label.setStyleSheet(
            f"color: {TEXT}; font-size: 15px; background: transparent;"
        )

        layout.addLayout(header)
        layout.addWidget(self.text_label)

        self.setStyleSheet(
            """
            QFrame#captionCard {
                background-color: rgba(255,255,255,8);
                border-radius: 12px;
                border: 1px solid rgba(255,255,255,15);
            }
            """
        )

    def set_caption(self, text: str, partial: bool) -> None:
        self.text_label.setText(text)
        self.state_label.setText("live" if partial else "")
        self.text_label.setStyleSheet(
            f"color: {'#d7dde5' if partial else TEXT}; "
            "font-size: 15px; background: transparent;"
        )


class SubtitleTab(QWidget):
    """Stage 1: English live subtitles only."""

    def __init__(self):
        super().__init__()

        self.bridge = SubtitleBridge()
        self.controller = None
        self._starting = False
        self._stopping = False

        self._entries: dict[tuple[str, int], CaptionCard] = {}
        self._entry_order: list[tuple[str, int]] = []
        self._max_entries = 60

        # History is private to the Subtitle tab/session.
        self._history: list[dict] = []
        self._history_index: dict[tuple[str, int], int] = {}

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 15, 20, 14)
        root.setSpacing(11)

        # Header inspired by the selected reference design, using old app colors.
        header = QHBoxLayout()
        header.setSpacing(8)

        icon = QLabel("🎙")
        icon.setStyleSheet(
            "font-size: 16px; background: transparent;"
        )

        title = QLabel("Meeting Subtitles")
        title.setStyleSheet(
            f"color: {TEXT}; font-size: 15px; font-weight: 750; "
            "background: transparent;"
        )

        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(
            f"color: {MUTED_DARK}; font-size: 9px; background: transparent;"
        )

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(
            f"color: {MUTED}; font-size: 12px; background: transparent;"
        )

        language = QLabel("English")
        language.setStyleSheet(
            f"color: {MUTED}; font-size: 11px; background: transparent;"
        )

        separator = QLabel("•")
        separator.setStyleSheet(
            f"color: {MUTED_DARK}; font-size: 10px; background: transparent;"
        )

        self.history_button = QPushButton("▣  Historial")
        self.history_button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )
        self.history_button.clicked.connect(self._show_history)
        self.history_button.setStyleSheet(_button_style())

        self.start_button = QPushButton("▶  Iniciar")
        self.start_button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )
        self.start_button.clicked.connect(self._toggle_session)
        self.start_button.setStyleSheet(_button_style(primary=True))

        header.addWidget(icon)
        header.addWidget(title)
        header.addSpacing(4)
        header.addWidget(self.status_dot)
        header.addWidget(self.status_label)
        header.addStretch()
        header.addWidget(language)
        header.addSpacing(6)
        header.addWidget(self.history_button)
        header.addWidget(self.start_button)

        root.addLayout(header)

        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(
            "background-color: rgba(255,255,255,18);"
        )
        root.addWidget(divider)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setStyleSheet(
            """
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 7px;
                margin: 4px 0 4px 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(255,255,255,45);
                border-radius: 3px;
                min-height: 25px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
            """
        )

        self.caption_host = QWidget()
        self.caption_host.setStyleSheet("background: transparent;")

        self.caption_layout = QVBoxLayout(self.caption_host)
        self.caption_layout.setContentsMargins(2, 2, 2, 2)
        self.caption_layout.setSpacing(8)

        self.empty_label = QLabel(
            "Inicia la reunión para ver los subtítulos en inglés aquí."
        )
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet(
            f"color: #718092; font-size: 13px; "
            "background: transparent; padding: 70px;"
        )

        self.caption_layout.addWidget(self.empty_label)
        self.caption_layout.addStretch()

        self.scroll_area.setWidget(self.caption_host)
        root.addWidget(self.scroll_area, 1)

        footer = QHBoxLayout()
        footer.setSpacing(14)

        self.mic_status = QLabel("🎤 Microphone: Ready")
        self.system_status = QLabel("🔊 System audio: Ready")
        self.engine_status = QLabel("⚡ Faster-Whisper: Local")

        for label in (
            self.mic_status,
            self.system_status,
            self.engine_status,
        ):
            label.setStyleSheet(
                f"color: {MUTED}; font-size: 10px; background: transparent;"
            )

        footer.addWidget(self.mic_status)
        footer.addWidget(self.system_status)
        footer.addStretch()
        footer.addWidget(self.engine_status)

        root.addLayout(footer)

    def _connect_signals(self) -> None:
        self.bridge.status_changed.connect(self._on_status_changed)
        self.bridge.subtitle_partial.connect(self._on_subtitle_partial)
        self.bridge.subtitle_ready.connect(self._on_subtitle_ready)

    def _toggle_session(self) -> None:
        if self._starting or self._stopping:
            return

        if self.controller is None:
            self._start_session()
        else:
            self._stop_session()

    def _start_session(self) -> None:
        self._starting = True
        self._clear_captions()
        self._clear_history()

        self.start_button.setEnabled(False)
        self.start_button.setText("⏳ Cargando...")
        self._on_status_changed("Cargando modelo local...")

        callbacks = {
            "on_status": lambda text: self.bridge.status_changed.emit(text),
            "on_subtitle_partial": (
                lambda source, text, segment_id:
                self.bridge.subtitle_partial.emit(source, text, segment_id)
            ),
            "on_subtitle": (
                lambda source, text, segment_id:
                self.bridge.subtitle_ready.emit(source, text, segment_id)
            ),
        }

        from vad_detector import ListenerController

        self.controller = ListenerController(callbacks)

        threading.Thread(
            target=self.controller.start,
            name="subtitle-controller",
            daemon=True,
        ).start()

    def _stop_session(self) -> None:
        controller = self.controller
        if controller is None:
            return

        self._stopping = True
        self.start_button.setEnabled(False)
        self.start_button.setText("⏳ Deteniendo...")

        def stop_worker() -> None:
            try:
                controller.stop()
            except Exception as exc:
                self.bridge.status_changed.emit(
                    f"Error stopping subtitles: {exc}"
                )

        threading.Thread(
            target=stop_worker,
            name="subtitle-stop",
            daemon=True,
        ).start()

    def _clear_captions(self) -> None:
        for card in self._entries.values():
            card.deleteLater()

        self._entries.clear()
        self._entry_order.clear()
        self.empty_label.show()

    def _clear_history(self) -> None:
        self._history.clear()
        self._history_index.clear()

    def _ensure_entry(
        self,
        source: str,
        segment_id: int,
    ) -> CaptionCard:
        key = (source, segment_id)

        existing = self._entries.get(key)
        if existing is not None:
            return existing

        self.empty_label.hide()

        card = CaptionCard(source, _now())
        insert_at = max(0, self.caption_layout.count() - 1)
        self.caption_layout.insertWidget(insert_at, card)

        self._entries[key] = card
        self._entry_order.append(key)

        while len(self._entry_order) > self._max_entries:
            old_key = self._entry_order.pop(0)
            old_card = self._entries.pop(old_key, None)
            if old_card is not None:
                old_card.deleteLater()

        return card

    def _on_subtitle_partial(
        self,
        source: str,
        text: str,
        segment_id: int,
    ) -> None:
        card = self._ensure_entry(source, segment_id)
        card.set_caption(text, partial=True)
        self._scroll_to_bottom()

    def _on_subtitle_ready(
        self,
        source: str,
        text: str,
        segment_id: int,
    ) -> None:
        key = (source, segment_id)

        card = self._ensure_entry(source, segment_id)
        card.set_caption(text, partial=False)

        history_item = {
            "key": key,
            "source": source,
            "time": _now(),
            "english": text,
        }

        existing_index = self._history_index.get(key)
        if existing_index is None:
            self._history_index[key] = len(self._history)
            self._history.append(history_item)
        else:
            self._history[existing_index] = history_item

        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        QTimer.singleShot(
            0,
            lambda: self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            ),
        )

    def _history_plain_text(self) -> str:
        if not self._history:
            return ""

        blocks = []
        for item in self._history:
            source = "YOU" if item["source"] == "YOU" else "MEETING"
            blocks.append(
                f"[{item['time']}] {source}\n{item['english']}"
            )

        return "\n\n".join(blocks)

    def _show_history(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Historial de la conversación")
        dialog.setMinimumSize(720, 520)
        dialog.setModal(True)
        dialog.setStyleSheet(
            f"""
            QDialog {{
                background-color: {APP_BG};
            }}
            QLabel {{
                background: transparent;
            }}
            """
        )

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()

        title = QLabel(
            f"Historial de la conversación · "
            f"{len(self._history)} intervenciones"
        )
        title.setStyleSheet(
            f"color: {TEXT}; font-size: 15px; font-weight: 750;"
        )

        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        history_view = QTextBrowser()
        history_view.setOpenExternalLinks(False)
        history_view.setStyleSheet(
            f"""
            QTextBrowser {{
                background-color: {CARD_BG};
                color: #e7ebef;
                border: 1px solid {BORDER};
                border-radius: 10px;
                padding: 12px;
                font-size: 13px;
            }}
            """
        )

        history_text = self._history_plain_text()
        if history_text:
            history_view.setPlainText(history_text)
        else:
            history_view.setPlainText(
                "Todavía no hay intervenciones en el historial."
            )

        layout.addWidget(history_view, 1)

        buttons = QHBoxLayout()
        buttons.addStretch()

        copy_button = QPushButton("📋 Copiar todo")
        copy_button.setStyleSheet(_button_style())
        copy_button.setEnabled(bool(history_text))

        def copy_history() -> None:
            QApplication.clipboard().setText(history_text)
            copy_button.setText("✓ Copiado")
            QTimer.singleShot(
                1200,
                lambda: copy_button.setText("📋 Copiar todo"),
            )

        copy_button.clicked.connect(copy_history)
        buttons.addWidget(copy_button)

        close_button = QPushButton("Cerrar")
        close_button.setStyleSheet(_button_style())
        close_button.clicked.connect(dialog.accept)
        buttons.addWidget(close_button)

        layout.addLayout(buttons)
        dialog.exec()

    def _on_status_changed(self, text: str) -> None:
        loading = text.startswith("Cargando")
        listening = text.startswith("Escuchando")
        stopped = text == "Detenido"
        error = text.startswith("Error")

        if loading:
            self._starting = True
            self.status_dot.setStyleSheet(
                f"color: {WARNING}; font-size: 9px; background: transparent;"
            )
            self.status_label.setText("Loading")
            return

        if listening:
            self._starting = False
            self._stopping = False

            self.status_dot.setStyleSheet(
                f"color: {GREEN}; font-size: 9px; background: transparent;"
            )
            self.status_label.setText("Listening")
            self.start_button.setText("■  Detener")
            self.start_button.setEnabled(True)

            mic_connected = "solo audio de reunión" not in text
            system_connected = "solo micrófono" not in text

            self.mic_status.setText(
                "🎤 Microphone: Connected"
                if mic_connected
                else "🎤 Microphone: Unavailable"
            )
            self.system_status.setText(
                "🔊 System audio: Connected"
                if system_connected
                else "🔊 System audio: Unavailable"
            )
            return

        if stopped:
            self.controller = None
            self._starting = False
            self._stopping = False

            self.status_dot.setStyleSheet(
                f"color: {MUTED_DARK}; font-size: 9px; background: transparent;"
            )
            self.status_label.setText("Ready")
            self.start_button.setText("▶  Iniciar")
            self.start_button.setEnabled(True)
            self.mic_status.setText("🎤 Microphone: Ready")
            self.system_status.setText("🔊 System audio: Ready")
            return

        if error:
            self.controller = None
            self._starting = False
            self._stopping = False

            self.status_dot.setStyleSheet(
                f"color: {RED}; font-size: 9px; background: transparent;"
            )
            self.status_label.setText("Error")
            self.start_button.setText("▶  Iniciar")
            self.start_button.setEnabled(True)
            self.engine_status.setText(
                "⚠ Faster-Whisper: check console"
            )
            return

        self.status_label.setText(text)

    def shutdown(self) -> None:
        controller = self.controller
        self.controller = None

        if controller is not None:
            try:
                controller.stop()
            except Exception:
                pass


class AssistantTab(QWidget):
    """Stage 2: independent technical interview copilot."""

    def __init__(self):
        super().__init__()

        self.bridge = AssistantBridge()
        self.controller = None
        self._starting = False
        self._stopping = False

        self._transcript_entries: dict[tuple[str, int], dict] = {}
        self._transcript_order: list[tuple[str, int]] = []
        self._max_transcript_entries = 30
        self._answer_buffer = ""
        self._pending_answer_buffer = ""
        self._answer_stream_started = False
        self._understood_question_text = ""
        self._pending_understood_question = ""
        self._pending_understood_topic = ""
        self._pending_understood_language = ""
        self._conversation_history: dict[tuple[str, int], dict] = {}
        self._has_any_turn = False
        self._has_interviewer_turn = False

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 15, 20, 16)
        root.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(8)

        icon = QLabel("✦")
        icon.setStyleSheet(
            f"color: {BLUE}; font-size: 16px; background: transparent;"
        )

        title = QLabel("Technical Interview Assistant")
        title.setStyleSheet(
            f"color: {TEXT}; font-size: 15px; font-weight: 750; "
            "background: transparent;"
        )

        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(
            f"color: {MUTED_DARK}; font-size: 9px; background: transparent;"
        )

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(
            f"color: {MUTED}; font-size: 11px; background: transparent;"
        )

        input_label = QLabel("Input: Auto")
        input_label.setToolTip(
            "Stage 2 detects English or Spanish independently for each turn."
        )
        input_label.setStyleSheet(
            f"color: {MUTED}; font-size: 11px; background: transparent;"
        )

        response_scope_label = QLabel("Responder a:")
        response_scope_label.setStyleSheet(
            f"color: {MUTED}; font-size: 11px; background: transparent;"
        )

        self.response_scope_combo = QComboBox()
        self.response_scope_combo.addItem(
            "Entrevistador",
            "interviewer",
        )
        self.response_scope_combo.addItem("Ambos", "both")
        self.response_scope_combo.setMinimumWidth(118)
        self.response_scope_combo.setToolTip(
            "Entrevistador: solo las preguntas del entrevistador generan "
            "respuestas. Ambos: YOU e INTERVIEWER pueden generarlas."
        )
        self.response_scope_combo.setStyleSheet(
            f"""
            QComboBox {{
                background-color: rgba(255,255,255,15);
                color: {TEXT};
                border: 1px solid rgba(255,255,255,25);
                border-radius: 8px;
                padding: 5px 9px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {CARD_BG_SOFT};
                color: {TEXT};
                selection-background-color: {BLUE};
            }}
            """
        )
        self.response_scope_combo.currentIndexChanged.connect(
            self._on_response_scope_changed
        )

        language_label = QLabel("Answer:")
        language_label.setStyleSheet(
            f"color: {MUTED}; font-size: 11px; background: transparent;"
        )

        self.language_combo = QComboBox()
        self.language_combo.addItem("English", "en")
        self.language_combo.addItem("Español", "es")
        self.language_combo.setMinimumWidth(105)
        self.language_combo.setStyleSheet(
            f"""
            QComboBox {{
                background-color: rgba(255,255,255,15);
                color: {TEXT};
                border: 1px solid rgba(255,255,255,25);
                border-radius: 8px;
                padding: 5px 9px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {CARD_BG_SOFT};
                color: {TEXT};
                selection-background-color: {BLUE};
            }}
            """
        )
        self.language_combo.currentIndexChanged.connect(
            self._on_language_changed
        )

        self.interview_button = QPushButton("▶  Iniciar entrevista")
        self.interview_button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )
        self.interview_button.setStyleSheet(_button_style(primary=True))
        self.interview_button.clicked.connect(self._toggle_session)

        header.addWidget(icon)
        header.addWidget(title)
        header.addSpacing(4)
        header.addWidget(self.status_dot)
        header.addWidget(self.status_label)
        header.addStretch()
        header.addWidget(input_label)
        header.addSpacing(6)
        header.addWidget(response_scope_label)
        header.addWidget(self.response_scope_combo)
        header.addSpacing(6)
        header.addWidget(language_label)
        header.addWidget(self.language_combo)
        header.addSpacing(6)
        header.addWidget(self.interview_button)

        root.addLayout(header)

        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(
            "background-color: rgba(255,255,255,18);"
        )
        root.addWidget(divider)

        conversation_header = QHBoxLayout()

        conversation_title = QLabel("Conversation")
        conversation_title.setStyleSheet(
            f"color: {TEXT}; font-size: 12px; font-weight: 700; "
            "background: transparent;"
        )

        self.activity_label = QLabel(
            "Waiting for interview audio"
        )
        self.activity_label.setStyleSheet(
            f"color: {MUTED}; font-size: 10px; background: transparent;"
        )

        self.copy_conversation_button = QPushButton(
            "📋 Copiar conversación"
        )
        self.copy_conversation_button.setStyleSheet(_button_style())
        self.copy_conversation_button.setEnabled(False)
        self.copy_conversation_button.clicked.connect(
            self._copy_conversation
        )

        conversation_header.addWidget(conversation_title)
        conversation_header.addStretch()
        conversation_header.addWidget(self.copy_conversation_button)
        conversation_header.addSpacing(6)
        conversation_header.addWidget(self.activity_label)

        root.addLayout(conversation_header)

        self.transcript_view = QTextBrowser()
        self.transcript_view.setOpenExternalLinks(False)
        self.transcript_view.setMinimumHeight(155)
        self.transcript_view.setStyleSheet(
            f"""
            QTextBrowser {{
                background-color: rgba(255,255,255,8);
                color: {TEXT};
                border: 1px solid rgba(255,255,255,15);
                border-radius: 12px;
                padding: 10px;
                font-size: 12px;
            }}
            """
        )
        self.transcript_view.setPlainText(
            "Start the interview to see YOU and INTERVIEWER here."
        )
        root.addWidget(self.transcript_view, 1)

        answer_header = QHBoxLayout()

        answer_title = QLabel("Suggested answer")
        answer_title.setStyleSheet(
            f"color: {TEXT}; font-size: 12px; font-weight: 700; "
            "background: transparent;"
        )

        self.answer_state = QLabel("Groq · GPT-OSS 120B")
        self.answer_state.setStyleSheet(
            f"color: {MUTED}; font-size: 10px; background: transparent;"
        )

        answer_header.addWidget(answer_title)
        answer_header.addStretch()
        answer_header.addWidget(self.answer_state)
        root.addLayout(answer_header)

        self.understood_question = QLabel(
            "Understood question: —"
        )
        self.understood_question.setWordWrap(True)
        self.understood_question.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.understood_question.setStyleSheet(
            f"color: {MUTED}; font-size: 11px; "
            "background: transparent; padding: 0 2px 2px 2px;"
        )
        root.addWidget(self.understood_question)

        self.answer_view = QTextBrowser()
        self.answer_view.setOpenExternalLinks(False)
        self.answer_view.setMinimumHeight(135)
        self.answer_view.setStyleSheet(
            f"""
            QTextBrowser {{
                background-color: {CARD_BG_SOFT};
                color: {TEXT};
                border: 1px solid rgba(77,166,255,45);
                border-radius: 12px;
                padding: 12px;
                font-size: 14px;
            }}
            """
        )
        self.answer_view.setPlainText(
            "A technical answer will appear here when a question is detected."
        )
        root.addWidget(self.answer_view)

        actions = QHBoxLayout()
        actions.setSpacing(8)

        self.answer_last_button = QPushButton("Responder último")
        self.answer_last_button.setStyleSheet(_button_style())
        self.answer_last_button.setEnabled(False)
        self.answer_last_button.clicked.connect(self._answer_last)

        self.regenerate_button = QPushButton("Regenerar")
        self.regenerate_button.setStyleSheet(_button_style())
        self.regenerate_button.setEnabled(False)
        self.regenerate_button.clicked.connect(self._regenerate)

        self.copy_question_button = QPushButton("📋 Copiar pregunta")
        self.copy_question_button.setStyleSheet(_button_style())
        self.copy_question_button.setEnabled(False)
        self.copy_question_button.clicked.connect(
            self._copy_question
        )

        self.copy_answer_button = QPushButton("📋 Copiar respuesta")
        self.copy_answer_button.setStyleSheet(_button_style())
        self.copy_answer_button.setEnabled(False)
        self.copy_answer_button.clicked.connect(self._copy_answer)

        actions.addWidget(self.answer_last_button)
        actions.addWidget(self.regenerate_button)
        actions.addWidget(self.copy_question_button)
        actions.addWidget(self.copy_answer_button)
        actions.addStretch()

        engine = QLabel("🎤 Groq Whisper Large V3   ⚡ Groq GPT-OSS 120B")
        engine.setStyleSheet(
            f"color: {MUTED}; font-size: 10px; background: transparent;"
        )
        actions.addWidget(engine)

        root.addLayout(actions)

    def _connect_signals(self) -> None:
        self.bridge.status_changed.connect(self._on_status_changed)
        self.bridge.transcript_partial.connect(
            self._on_transcript_partial
        )
        self.bridge.transcript_final.connect(
            self._on_transcript_final
        )
        self.bridge.transcript_rejected.connect(
            self._on_transcript_rejected
        )
        self.bridge.question_candidate.connect(
            self._on_question_candidate
        )
        self.bridge.turn_understood.connect(
            self._on_turn_understood
        )
        self.bridge.question_waiting.connect(
            self._on_question_waiting
        )
        self.bridge.question_detected.connect(
            self._on_question_detected
        )
        self.bridge.question_ignored.connect(
            self._on_question_ignored
        )
        self.bridge.answer_started.connect(
            self._on_answer_started
        )
        self.bridge.answer_delta.connect(self._on_answer_delta)
        self.bridge.answer_completed.connect(
            self._on_answer_completed
        )
        self.bridge.assistant_error.connect(
            self._on_assistant_error
        )

    def _toggle_session(self) -> None:
        if self._starting or self._stopping:
            return

        if self.controller is None:
            self._start_session()
        else:
            self._stop_session()

    def _start_session(self) -> None:
        self._starting = True
        self._clear_session_ui()

        self.interview_button.setEnabled(False)
        self.interview_button.setText("⏳ Cargando...")
        self._on_status_changed("Preparando audio...")

        callbacks = {
            "on_status": (
                lambda text: self.bridge.status_changed.emit(text)
            ),
            "on_transcript_partial": (
                lambda speaker, text, segment_id:
                self.bridge.transcript_partial.emit(
                    speaker,
                    text,
                    segment_id,
                )
            ),
            "on_transcript_final": (
                lambda speaker, text, segment_id:
                self.bridge.transcript_final.emit(
                    speaker,
                    text,
                    segment_id,
                )
            ),
            "on_transcript_rejected": (
                lambda speaker, text:
                self.bridge.transcript_rejected.emit(
                    speaker,
                    text,
                )
            ),
            "on_question_candidate": (
                lambda text:
                self.bridge.question_candidate.emit(text)
            ),
            "on_turn_understood": (
                lambda question, topic, language:
                self.bridge.turn_understood.emit(
                    question,
                    topic,
                    language,
                )
            ),
            "on_question_waiting": (
                lambda text:
                self.bridge.question_waiting.emit(text)
            ),
            "on_question_detected": (
                lambda text:
                self.bridge.question_detected.emit(text)
            ),
            "on_question_ignored": (
                lambda text:
                self.bridge.question_ignored.emit(text)
            ),
            "on_answer_started": (
                lambda text:
                self.bridge.answer_started.emit(text)
            ),
            "on_answer_delta": (
                lambda text:
                self.bridge.answer_delta.emit(text)
            ),
            "on_answer_completed": (
                lambda text:
                self.bridge.answer_completed.emit(text)
            ),
            "on_assistant_error": (
                lambda text:
                self.bridge.assistant_error.emit(text)
            ),
        }

        from interview_controller import InterviewController

        language = self.language_combo.currentData() or "en"
        response_scope = (
            self.response_scope_combo.currentData()
            or "interviewer"
        )
        self.controller = InterviewController(
            callbacks=callbacks,
            language=language,
            response_scope=response_scope,
        )

        threading.Thread(
            target=self.controller.start,
            name="interview-controller",
            daemon=True,
        ).start()

    def _stop_session(self) -> None:
        controller = self.controller
        if controller is None:
            return

        self._stopping = True
        self.interview_button.setEnabled(False)
        self.interview_button.setText("⏳ Deteniendo...")

        def stop_worker() -> None:
            try:
                controller.stop()
            except Exception as exc:
                self.bridge.assistant_error.emit(str(exc))
                self.bridge.status_changed.emit("Detenido")

        threading.Thread(
            target=stop_worker,
            name="interview-stop",
            daemon=True,
        ).start()

    def _clear_session_ui(self) -> None:
        self._transcript_entries.clear()
        self._transcript_order.clear()
        self._conversation_history.clear()
        self._answer_buffer = ""
        self._pending_answer_buffer = ""
        self._answer_stream_started = False
        self._understood_question_text = ""
        self._pending_understood_question = ""
        self._pending_understood_topic = ""
        self._pending_understood_language = ""
        self._has_any_turn = False
        self._has_interviewer_turn = False

        self.transcript_view.setPlainText(
            "Listening for YOU and INTERVIEWER..."
        )
        self.answer_view.setPlainText(
            "Waiting for a technical question..."
        )
        self.activity_label.setText("Waiting for interview audio")
        self.answer_state.setText("Groq · GPT-OSS 120B")
        self.understood_question.setText("Understood question: —")
        self.answer_last_button.setEnabled(False)
        self.regenerate_button.setEnabled(False)
        self.copy_conversation_button.setEnabled(False)
        self.copy_question_button.setEnabled(False)
        self.copy_answer_button.setEnabled(False)

    def _on_language_changed(self, _index: int = -1) -> None:
        controller = self.controller
        if controller is None:
            return

        language = self.language_combo.currentData() or "en"
        controller.set_language(language)

    def _on_response_scope_changed(self, _index: int = -1) -> None:
        scope = (
            self.response_scope_combo.currentData()
            or "interviewer"
        )

        if self.controller is not None:
            self.controller.set_response_scope(scope)

        if self.status_label.text() == "Listening":
            self.activity_label.setText(
                "Listening for both speakers"
                if scope == "both"
                else "Listening for interviewer"
            )

        self._refresh_answer_last_button()

    def _refresh_answer_last_button(self) -> None:
        scope = (
            self.response_scope_combo.currentData()
            or "interviewer"
        )
        if scope == "both":
            enabled = self._has_any_turn
        else:
            enabled = self._has_interviewer_turn

        self.answer_last_button.setEnabled(enabled)

    def _update_transcript(
        self,
        speaker: str,
        text: str,
        segment_id: int,
        partial: bool,
    ) -> None:
        key = (speaker, segment_id)

        if key not in self._transcript_entries:
            self._transcript_order.append(key)

        self._transcript_entries[key] = {
            "speaker": speaker,
            "text": text,
            "partial": partial,
        }

        while len(self._transcript_order) > self._max_transcript_entries:
            old_key = self._transcript_order.pop(0)
            self._transcript_entries.pop(old_key, None)

        blocks = []
        for entry_key in self._transcript_order:
            entry = self._transcript_entries.get(entry_key)
            if entry is None:
                continue

            state = " · live" if entry["partial"] else ""
            blocks.append(
                f'{entry["speaker"]}{state}\n{entry["text"]}'
            )

        self.transcript_view.setPlainText("\n\n".join(blocks))
        scrollbar = self.transcript_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_transcript_partial(
        self,
        speaker: str,
        text: str,
        segment_id: int,
    ) -> None:
        self._update_transcript(
            speaker,
            text,
            segment_id,
            partial=True,
        )

    def _on_transcript_final(
        self,
        speaker: str,
        text: str,
        segment_id: int,
    ) -> None:
        self._update_transcript(
            speaker,
            text,
            segment_id,
            partial=False,
        )

        key = (speaker, segment_id)
        existing = self._conversation_history.get(key)
        timestamp = (
            existing["timestamp"]
            if existing is not None
            else _now()
        )
        self._conversation_history[key] = {
            "timestamp": timestamp,
            "speaker": speaker,
            "text": text,
        }
        self.copy_conversation_button.setEnabled(
            bool(self._conversation_history)
        )

        self._has_any_turn = True
        if speaker == "INTERVIEWER":
            self._has_interviewer_turn = True

        self._refresh_answer_last_button()

    def _on_transcript_rejected(
        self,
        speaker: str,
        text: str,
    ) -> None:
        self.activity_label.setText(
            "Noisy speech ignored"
        )

    def _on_question_candidate(self, text: str) -> None:
        # Keep the last confirmed answer visible while this turn is only a
        # candidate. It may still become WAIT or IGNORE.
        self._pending_understood_question = ""
        self._pending_understood_topic = ""
        self._pending_understood_language = ""
        self.activity_label.setText(
            "Analyzing question..."
        )

    def _on_turn_understood(
        self,
        question: str,
        topic: str,
        language: str,
    ) -> None:
        # The contextualizer runs before WAIT / IGNORE / ANSWER is known.
        # Keep this result pending so non-technical turns do not replace the
        # last confirmed technical question in the UI.
        self._pending_understood_question = question.strip()
        self._pending_understood_topic = topic.strip()
        self._pending_understood_language = language.strip()

    def _on_question_waiting(self, text: str) -> None:
        self._pending_understood_question = ""
        self._pending_understood_topic = ""
        self._pending_understood_language = ""
        self.activity_label.setText(
            "Waiting for the question to finish..."
        )

    def _on_question_detected(self, text: str) -> None:
        question = (
            self._pending_understood_question
            or text.strip()
        )
        topic = self._pending_understood_topic
        language = self._pending_understood_language

        details = []
        if topic:
            details.append(topic)
        if language and language != "unknown":
            details.append(language.upper())

        suffix = f"  ·  {' · '.join(details)}" if details else ""
        self._understood_question_text = question
        self.understood_question.setText(
            f"Understood question: {question}{suffix}"
        )
        self.copy_question_button.setEnabled(bool(question))

        self._pending_understood_question = ""
        self._pending_understood_topic = ""
        self._pending_understood_language = ""

        self.activity_label.setText(
            "Technical question detected"
        )

    def _on_question_ignored(self, text: str) -> None:
        self._pending_understood_question = ""
        self._pending_understood_topic = ""
        self._pending_understood_language = ""
        self.activity_label.setText(
            "Non-technical turn ignored"
        )

    def _on_answer_started(self, question: str) -> None:
        # Keep the previous confirmed answer visible until the new generation
        # produces actual text. This prevents an empty/error response from
        # leaving the interviewee with a blank panel.
        self._pending_answer_buffer = ""
        self._answer_stream_started = False
        self.answer_state.setText("Generating...")
        self.regenerate_button.setEnabled(False)
        self.copy_answer_button.setEnabled(bool(self._answer_buffer))

    def _on_answer_delta(self, delta: str) -> None:
        if not delta:
            return

        if not self._answer_stream_started:
            self._answer_stream_started = True
            self._pending_answer_buffer = ""
            self.answer_view.clear()

        self._pending_answer_buffer += delta
        self.answer_view.setPlainText(self._pending_answer_buffer)

    def _on_answer_completed(self, answer: str) -> None:
        completed = (answer or self._pending_answer_buffer).strip()

        if completed:
            self._answer_buffer = completed
            self.answer_view.setPlainText(completed)
            self.answer_state.setText("Ready")
            self.regenerate_button.setEnabled(True)
            self.copy_answer_button.setEnabled(True)
            self.activity_label.setText("Answer ready")
        else:
            # No content arrived. Preserve the previous confirmed answer.
            self.answer_state.setText(
                "Previous answer"
                if self._answer_buffer
                else "No answer"
            )
            self.regenerate_button.setEnabled(bool(self._answer_buffer))
            self.copy_answer_button.setEnabled(bool(self._answer_buffer))
            self.activity_label.setText(
                "No new answer generated"
            )

        self._pending_answer_buffer = ""
        self._answer_stream_started = False

    def _on_assistant_error(self, text: str) -> None:
        # Preserve a useful previous answer when a new generation fails.
        self._pending_answer_buffer = ""
        self._answer_stream_started = False
        self.answer_state.setText("Error")
        self.regenerate_button.setEnabled(bool(self._answer_buffer))
        self.copy_answer_button.setEnabled(bool(self._answer_buffer))

        if not self._answer_buffer:
            self.answer_view.setPlainText(f"Error: {text}")

        self.activity_label.setText(
            f"Assistant error: {text}"
        )

    def _answer_last(self) -> None:
        if self.controller is not None:
            self.controller.answer_last_interviewer_turn()

    def _regenerate(self) -> None:
        if self.controller is not None:
            self.controller.regenerate()

    def _copy_conversation(self) -> None:
        if not self._conversation_history:
            return

        blocks = []
        for entry in self._conversation_history.values():
            blocks.append(
                f'[{entry["timestamp"]}] {entry["speaker"]}\n'
                f'{entry["text"]}'
            )

        QApplication.clipboard().setText(
            "\n\n".join(blocks)
        )
        self.copy_conversation_button.setText(
            "✓ Conversación copiada"
        )
        QTimer.singleShot(
            1200,
            lambda: self.copy_conversation_button.setText(
                "📋 Copiar conversación"
            ),
        )

    def _copy_question(self) -> None:
        if not self._understood_question_text:
            return

        QApplication.clipboard().setText(
            self._understood_question_text
        )
        self.copy_question_button.setText("✓ Pregunta copiada")
        QTimer.singleShot(
            1200,
            lambda: self.copy_question_button.setText(
                "📋 Copiar pregunta"
            ),
        )

    def _copy_answer(self) -> None:
        if not self._answer_buffer:
            return

        QApplication.clipboard().setText(self._answer_buffer)
        self.copy_answer_button.setText("✓ Respuesta copiada")
        QTimer.singleShot(
            1200,
            lambda: self.copy_answer_button.setText(
                "📋 Copiar respuesta"
            ),
        )

    def _on_status_changed(self, text: str) -> None:
        loading = text.startswith("Cargando")
        listening = text.startswith("Escuchando")
        stopped = text == "Detenido"
        error = text.startswith("Error")

        if loading:
            self._starting = True
            self.status_dot.setStyleSheet(
                f"color: {WARNING}; font-size: 9px; background: transparent;"
            )
            self.status_label.setText("Loading")
            return

        if listening:
            self._starting = False
            self._stopping = False
            self.status_dot.setStyleSheet(
                f"color: {GREEN}; font-size: 9px; background: transparent;"
            )
            self.status_label.setText("Listening")
            self.interview_button.setText("■  Detener")
            self.interview_button.setEnabled(True)
            scope = (
                self.response_scope_combo.currentData()
                or "interviewer"
            )
            self.activity_label.setText(
                "Listening for both speakers"
                if scope == "both"
                else "Listening for interviewer"
            )
            return

        if stopped:
            self.controller = None
            self._starting = False
            self._stopping = False
            self.status_dot.setStyleSheet(
                f"color: {MUTED_DARK}; font-size: 9px; background: transparent;"
            )
            self.status_label.setText("Ready")
            self.interview_button.setText("▶  Iniciar entrevista")
            self.interview_button.setEnabled(True)
            self.activity_label.setText("Stopped")
            return

        if error:
            self.controller = None
            self._starting = False
            self._stopping = False
            self.status_dot.setStyleSheet(
                f"color: {RED}; font-size: 9px; background: transparent;"
            )
            self.status_label.setText("Error")
            self.interview_button.setText("▶  Iniciar entrevista")
            self.interview_button.setEnabled(True)
            self.answer_view.setPlainText(text)
            return

        self.status_label.setText(text)

    def shutdown(self) -> None:
        controller = self.controller
        self.controller = None

        if controller is not None:
            try:
                controller.stop()
            except Exception:
                pass


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self._drag_pos = None
        self._build_window()
        self._build_ui()

    def _build_window(self) -> None:
        self.setWindowTitle("Meeting Assistant")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground
        )
        self.setMinimumSize(760, 500)
        self.resize(900, 560)
        self.move(60, 60)

    def _build_ui(self) -> None:
        shell = QFrame()
        shell.setObjectName("mainCard")
        shell.setStyleSheet(
            """
            QFrame#mainCard {
                background-color: rgba(15, 18, 22, 242);
                border-radius: 16px;
                border: 1px solid rgba(255,255,255,20);
            }
            QLabel {
                background: transparent;
            }
            """
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(shell)

        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        # Window/title row.
        title_bar = QHBoxLayout()
        title_bar.setContentsMargins(18, 10, 10, 4)
        title_bar.setSpacing(8)

        app_title = QLabel("Meeting Assistant")
        app_title.setStyleSheet(
            f"color: {TEXT}; font-size: 13px; font-weight: 750;"
        )

        always_on_top = QLabel("📌 Always on top")
        always_on_top.setStyleSheet(
            f"color: {MUTED}; font-size: 10px;"
        )

        close_button = QPushButton("×")
        close_button.setFixedSize(30, 30)
        close_button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )
        close_button.clicked.connect(self.close)
        close_button.setStyleSheet(
            """
            QPushButton {
                background: transparent;
                color: #8f9baa;
                border: none;
                border-radius: 8px;
                font-size: 19px;
                font-weight: 700;
            }
            QPushButton:hover {
                background-color: rgba(255,80,80,50);
                color: white;
            }
            """
        )

        title_bar.addWidget(app_title)
        title_bar.addStretch()
        title_bar.addWidget(always_on_top)
        title_bar.addWidget(close_button)

        shell_layout.addLayout(title_bar)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setStyleSheet(
            f"""
            QTabWidget::pane {{
                border: none;
                background: transparent;
            }}
            QTabBar {{
                background: transparent;
            }}
            QTabBar::tab {{
                background: transparent;
                color: {MUTED};
                border: none;
                padding: 9px 18px;
                margin-left: 8px;
                font-size: 12px;
                font-weight: 650;
            }}
            QTabBar::tab:selected {{
                color: {TEXT};
                border-bottom: 2px solid {BLUE};
            }}
            QTabBar::tab:hover {{
                color: {TEXT};
            }}
            """
        )

        self.subtitle_tab = SubtitleTab()
        self.assistant_tab = AssistantTab()

        self.tabs.addTab(self.subtitle_tab, "Subtítulo")
        self.tabs.addTab(self.assistant_tab, "Asistente")

        # Stage 1 is always selected when the app starts.
        self.tabs.setCurrentIndex(0)

        shell_layout.addWidget(self.tabs, 1)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = (
                event.globalPosition().toPoint()
                - self.frameGeometry().topLeft()
            )
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if (
            self._drag_pos is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self.move(
                event.globalPosition().toPoint() - self._drag_pos
            )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def closeEvent(self, event) -> None:
        self.subtitle_tab.shutdown()
        self.assistant_tab.shutdown()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
