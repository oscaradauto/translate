"""Desktop UI with two independent meeting modes.

Tab 1 - Subtítulo:
    English-only live captions for a multi-person meeting. Uses Faster-Whisper
    locally and never invokes the assistant.

Tab 2 - Asistente:
    Separate technical-interview UI prepared for Stage 2 / OpenAI.
"""

from __future__ import annotations

import sys
import threading
import time

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


BG = "#0d1117"
PANEL = "#131922"
PANEL_SOFT = "#171e28"
BORDER = "#273140"
TEXT = "#f2f5f8"
MUTED = "#8b98a8"
BLUE = "#4b8cff"
GREEN = "#35c477"
ORANGE = "#f5a24a"
RED = "#ef6461"


def _now() -> str:
    return time.strftime("%I:%M:%S %p")


class SubtitleBridge(QObject):
    status_changed = pyqtSignal(str)
    subtitle_partial = pyqtSignal(str, str, int)
    subtitle_ready = pyqtSignal(str, str, int)


class CaptionRow(QFrame):
    """One continuously updated caption block."""

    def __init__(self, source: str, timestamp: str):
        super().__init__()
        self.setObjectName("captionRow")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 9, 4, 12)
        layout.setSpacing(5)

        header = QHBoxLayout()
        header.setSpacing(8)

        source_name = "You" if source == "YOU" else "Meeting"
        self.source_label = QLabel(source_name)
        self.source_label.setStyleSheet(
            f"color: {BLUE if source == 'YOU' else ORANGE}; "
            "font-size: 12px; font-weight: 700; background: transparent;"
        )

        self.time_label = QLabel(timestamp)
        self.time_label.setStyleSheet(
            f"color: {MUTED}; font-size: 10px; background: transparent;"
        )

        header.addWidget(self.source_label)
        header.addStretch()
        header.addWidget(self.time_label)

        self.text_label = QLabel("")
        self.text_label.setWordWrap(True)
        self.text_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.text_label.setStyleSheet(
            f"color: {TEXT}; font-size: 16px; line-height: 1.4; "
            "background: transparent;"
        )

        layout.addLayout(header)
        layout.addWidget(self.text_label)

        self.setStyleSheet(
            f"""
            QFrame#captionRow {{
                background: transparent;
                border: none;
                border-bottom: 1px solid {BORDER};
            }}
            """
        )

    def set_caption(self, text: str, partial: bool) -> None:
        self.text_label.setText(text)
        color = "#d7dde5" if partial else TEXT
        self.text_label.setStyleSheet(
            f"color: {color}; font-size: 16px; background: transparent;"
        )


class SubtitleTab(QWidget):
    """Stage 1. English live subtitles only."""

    def __init__(self):
        super().__init__()

        self.bridge = SubtitleBridge()
        self.controller = None
        self._starting = False
        self._stopping = False
        self._entries: dict[tuple[str, int], CaptionRow] = {}
        self._entry_order: list[tuple[str, int]] = []
        self._max_entries = 40

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 18)
        root.setSpacing(16)

        heading = QHBoxLayout()

        titles = QVBoxLayout()
        titles.setSpacing(2)

        title = QLabel("Live subtitles")
        title.setStyleSheet(
            f"color: {TEXT}; font-size: 20px; font-weight: 750;"
        )

        subtitle = QLabel(
            "English meeting · local transcription · approximately 6 participants"
        )
        subtitle.setStyleSheet(
            f"color: {MUTED}; font-size: 11px;"
        )

        titles.addWidget(title)
        titles.addWidget(subtitle)

        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(
            f"color: {MUTED}; font-size: 10px;"
        )
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(
            f"color: {MUTED}; font-size: 12px;"
        )

        self.start_button = QPushButton("Start subtitles")
        self.start_button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )
        self.start_button.setMinimumHeight(34)
        self.start_button.clicked.connect(self._toggle_session)
        self.start_button.setStyleSheet(
            f"""
            QPushButton {{
                background: {BLUE};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 7px 16px;
                font-weight: 700;
            }}
            QPushButton:hover {{ background: #5a97ff; }}
            QPushButton:disabled {{
                background: #2b3442;
                color: #6f7b89;
            }}
            """
        )

        heading.addLayout(titles)
        heading.addStretch()
        heading.addWidget(self.status_dot)
        heading.addWidget(self.status_label)
        heading.addSpacing(10)
        heading.addWidget(self.start_button)

        root.addLayout(heading)

        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background: {BORDER};")
        root.addWidget(divider)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setStyleSheet(
            f"""
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 7px;
            }}
            QScrollBar::handle:vertical {{
                background: #354153;
                border-radius: 3px;
                min-height: 28px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            """
        )

        self.caption_host = QWidget()
        self.caption_host.setStyleSheet("background: transparent;")
        self.caption_layout = QVBoxLayout(self.caption_host)
        self.caption_layout.setContentsMargins(8, 2, 8, 2)
        self.caption_layout.setSpacing(0)

        self.empty_label = QLabel(
            "Start the meeting to see English captions here."
        )
        self.empty_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.empty_label.setStyleSheet(
            f"color: {MUTED}; font-size: 13px; padding: 80px;"
        )

        self.caption_layout.addWidget(self.empty_label)
        self.caption_layout.addStretch()

        self.scroll_area.setWidget(self.caption_host)
        root.addWidget(self.scroll_area, 1)

        footer = QHBoxLayout()
        footer.setSpacing(18)

        self.mic_status = QLabel("🎤 Microphone: Ready")
        self.system_status = QLabel("🔊 Meeting audio: Ready")
        self.engine_status = QLabel("⚡ Faster-Whisper: Local")

        for label in (
            self.mic_status,
            self.system_status,
            self.engine_status,
        ):
            label.setStyleSheet(
                f"color: {MUTED}; font-size: 10px;"
            )

        footer.addWidget(self.mic_status)
        footer.addWidget(self.system_status)
        footer.addStretch()
        footer.addWidget(self.engine_status)
        root.addLayout(footer)

    def _connect_signals(self) -> None:
        self.bridge.status_changed.connect(
            self._on_status_changed
        )
        self.bridge.subtitle_partial.connect(
            self._on_subtitle_partial
        )
        self.bridge.subtitle_ready.connect(
            self._on_subtitle_ready
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
        self._clear_captions()
        self.start_button.setEnabled(False)
        self.start_button.setText("Loading model…")
        self._on_status_changed("Cargando modelo local...")

        callbacks = {
            "on_status": lambda text: (
                self.bridge.status_changed.emit(text)
            ),
            "on_subtitle_partial": lambda source, text, segment_id: (
                self.bridge.subtitle_partial.emit(
                    source,
                    text,
                    segment_id,
                )
            ),
            "on_subtitle": lambda source, text, segment_id: (
                self.bridge.subtitle_ready.emit(
                    source,
                    text,
                    segment_id,
                )
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
        self.start_button.setText("Stopping…")

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
        for row in self._entries.values():
            row.deleteLater()

        self._entries.clear()
        self._entry_order.clear()
        self.empty_label.show()

    def _ensure_entry(
        self,
        source: str,
        segment_id: int,
    ) -> CaptionRow:
        key = (source, segment_id)
        existing = self._entries.get(key)
        if existing is not None:
            return existing

        self.empty_label.hide()

        row = CaptionRow(source, _now())
        insert_at = max(
            0,
            self.caption_layout.count() - 1,
        )
        self.caption_layout.insertWidget(insert_at, row)

        self._entries[key] = row
        self._entry_order.append(key)

        while len(self._entry_order) > self._max_entries:
            old_key = self._entry_order.pop(0)
            old = self._entries.pop(old_key, None)
            if old is not None:
                old.deleteLater()

        return row

    def _on_subtitle_partial(
        self,
        source: str,
        text: str,
        segment_id: int,
    ) -> None:
        row = self._ensure_entry(source, segment_id)
        row.set_caption(text, partial=True)
        self._scroll_to_bottom()

    def _on_subtitle_ready(
        self,
        source: str,
        text: str,
        segment_id: int,
    ) -> None:
        row = self._ensure_entry(source, segment_id)
        row.set_caption(text, partial=False)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        QTimer.singleShot(
            0,
            lambda: self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
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
                f"color: {ORANGE}; font-size: 10px;"
            )
            self.status_label.setText("Loading local model")
            return

        if listening:
            self._starting = False
            self._stopping = False
            self.status_dot.setStyleSheet(
                f"color: {GREEN}; font-size: 10px;"
            )
            self.status_label.setText("Listening")
            self.start_button.setText("Stop subtitles")
            self.start_button.setEnabled(True)

            mic_connected = "solo audio de reunión" not in text
            meeting_connected = "solo micrófono" not in text

            self.mic_status.setText(
                "🎤 Microphone: Connected"
                if mic_connected
                else "🎤 Microphone: Unavailable"
            )
            self.system_status.setText(
                "🔊 Meeting audio: Connected"
                if meeting_connected
                else "🔊 Meeting audio: Unavailable"
            )
            return

        if stopped:
            self.controller = None
            self._starting = False
            self._stopping = False
            self.status_dot.setStyleSheet(
                f"color: {MUTED}; font-size: 10px;"
            )
            self.status_label.setText("Ready")
            self.start_button.setText("Start subtitles")
            self.start_button.setEnabled(True)
            self.mic_status.setText("🎤 Microphone: Ready")
            self.system_status.setText(
                "🔊 Meeting audio: Ready"
            )
            return

        if error:
            self.controller = None
            self._starting = False
            self._stopping = False
            self.status_dot.setStyleSheet(
                f"color: {RED}; font-size: 10px;"
            )
            self.status_label.setText("Error")
            self.start_button.setText("Start subtitles")
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
    """Independent Stage 2 screen. OpenAI wiring is intentionally separate."""

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(18)

        heading = QHBoxLayout()

        titles = QVBoxLayout()
        titles.setSpacing(2)

        title = QLabel("Technical interview assistant")
        title.setStyleSheet(
            f"color: {TEXT}; font-size: 20px; font-weight: 750;"
        )
        subtitle = QLabel(
            "Independent interview session · 2–3 participants · OpenAI"
        )
        subtitle.setStyleSheet(
            f"color: {MUTED}; font-size: 11px;"
        )

        titles.addWidget(title)
        titles.addWidget(subtitle)

        language_label = QLabel("Answer language")
        language_label.setStyleSheet(
            f"color: {MUTED}; font-size: 10px;"
        )

        self.language_combo = QComboBox()
        self.language_combo.addItem("English", "en")
        self.language_combo.addItem("Español", "es")
        self.language_combo.setMinimumWidth(110)
        self.language_combo.setStyleSheet(
            f"""
            QComboBox {{
                background: {PANEL_SOFT};
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 7px;
                padding: 6px 10px;
            }}
            QComboBox QAbstractItemView {{
                background: {PANEL_SOFT};
                color: {TEXT};
                selection-background-color: {BLUE};
            }}
            """
        )

        lang_box = QVBoxLayout()
        lang_box.setSpacing(3)
        lang_box.addWidget(language_label)
        lang_box.addWidget(self.language_combo)

        heading.addLayout(titles)
        heading.addStretch()
        heading.addLayout(lang_box)
        root.addLayout(heading)

        info = QFrame()
        info.setStyleSheet(
            f"""
            QFrame {{
                background: {PANEL_SOFT};
                border: 1px solid {BORDER};
                border-radius: 12px;
            }}
            """
        )
        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(18, 16, 18, 16)
        info_layout.setSpacing(7)

        info_title = QLabel("Stage 2 · OpenAI assistant")
        info_title.setStyleSheet(
            f"color: {TEXT}; font-size: 14px; font-weight: 700;"
        )
        info_text = QLabel(
            "This screen is intentionally independent from Live Subtitles. "
            "The next Stage 2 change will add interview transcription, "
            "conversation context, technical-question detection and streamed "
            "answers in the selected language."
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet(
            f"color: {MUTED}; font-size: 12px;"
        )

        info_layout.addWidget(info_title)
        info_layout.addWidget(info_text)
        root.addWidget(info)

        conversation_title = QLabel("Conversation")
        conversation_title.setStyleSheet(
            f"color: {TEXT}; font-size: 13px; font-weight: 700;"
        )
        root.addWidget(conversation_title)

        conversation = QFrame()
        conversation.setMinimumHeight(130)
        conversation.setStyleSheet(
            f"""
            QFrame {{
                background: {PANEL};
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
            """
        )
        conversation_layout = QVBoxLayout(conversation)
        conversation_placeholder = QLabel(
            "Interview transcript will appear here."
        )
        conversation_placeholder.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        conversation_placeholder.setStyleSheet(
            f"color: {MUTED}; font-size: 12px;"
        )
        conversation_layout.addWidget(conversation_placeholder)
        root.addWidget(conversation)

        answer_title = QLabel("Suggested answer")
        answer_title.setStyleSheet(
            f"color: {TEXT}; font-size: 13px; font-weight: 700;"
        )
        root.addWidget(answer_title)

        answer = QFrame()
        answer.setMinimumHeight(120)
        answer.setStyleSheet(
            f"""
            QFrame {{
                background: {PANEL};
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
            """
        )
        answer_layout = QVBoxLayout(answer)
        answer_placeholder = QLabel(
            "OpenAI response will stream here after a technical "
            "question is detected."
        )
        answer_placeholder.setWordWrap(True)
        answer_placeholder.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        answer_placeholder.setStyleSheet(
            f"color: {MUTED}; font-size: 12px;"
        )
        answer_layout.addWidget(answer_placeholder)
        root.addWidget(answer)

        root.addStretch()


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
        self.setMinimumSize(760, 520)
        self.resize(920, 640)
        self.move(55, 55)

    def _build_ui(self) -> None:
        shell = QFrame()
        shell.setObjectName("shell")
        shell.setStyleSheet(
            f"""
            QFrame#shell {{
                background: {BG};
                border: 1px solid {BORDER};
                border-radius: 14px;
            }}
            QLabel {{
                background: transparent;
            }}
            """
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(shell)

        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        title_bar = QHBoxLayout()
        title_bar.setContentsMargins(20, 12, 12, 8)

        title = QLabel("Meeting Assistant")
        title.setStyleSheet(
            f"color: {TEXT}; font-size: 13px; font-weight: 700;"
        )
        pin = QLabel("📌 Always on top")
        pin.setStyleSheet(
            f"color: {MUTED}; font-size: 10px;"
        )

        close_button = QPushButton("×")
        close_button.setFixedSize(30, 30)
        close_button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )
        close_button.clicked.connect(self.close)
        close_button.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                color: {MUTED};
                border: none;
                border-radius: 7px;
                font-size: 19px;
            }}
            QPushButton:hover {{
                background: #392126;
                color: white;
            }}
            """
        )

        title_bar.addWidget(title)
        title_bar.addStretch()
        title_bar.addWidget(pin)
        title_bar.addSpacing(8)
        title_bar.addWidget(close_button)
        shell_layout.addLayout(title_bar)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setStyleSheet(
            f"""
            QTabWidget::pane {{
                border: none;
                background: {BG};
            }}
            QTabBar {{
                background: {BG};
            }}
            QTabBar::tab {{
                background: transparent;
                color: {MUTED};
                border: none;
                padding: 10px 20px;
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

        # Stage 1 is always the default screen.
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
                event.globalPosition().toPoint()
                - self._drag_pos
            )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def closeEvent(self, event) -> None:
        self.subtitle_tab.shutdown()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
