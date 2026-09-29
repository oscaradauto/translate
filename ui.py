import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import threading
import time

from PyQt6.QtCore import Qt, QObject, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config import get_meeting_language, set_meeting_language
from vad_detector import ListenerController


PILL_STYLE = """
QPushButton, QComboBox {
    background-color: rgba(255,255,255,15);
    color: white;
    border-radius: 16px;
    padding: 6px 14px;
    border: 1px solid rgba(255,255,255,25);
}
QComboBox::drop-down {
    border: none;
    width: 18px;
}
QPushButton:hover, QComboBox:hover {
    background-color: rgba(255,255,255,25);
}
QComboBox QAbstractItemView {
    background-color: #1e1e1e;
    color: white;
    selection-background-color: #3a3a5a;
    selection-color: white;
    border: 1px solid rgba(255,255,255,30);
    outline: none;
}
"""

START_BTN_STYLE = """
QPushButton {
    background-color: rgba(255,255,255,15);
    color: white;
    border-radius: 16px;
    padding: 6px 16px;
    border: 1px solid rgba(255,255,255,25);
    font-weight: bold;
}
QPushButton:hover {
    background-color: rgba(255,255,255,25);
}
"""

CLOSE_BTN_STYLE = """
QPushButton {
    background-color: rgba(255,255,255,15);
    color: white;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,25);
    font-size: 14px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: rgba(255,80,80,50);
}
"""

SOURCE_COLORS = {
    "YOU": "#4da6ff",
    "COMPANION": "#ffa64d",
}


def _now():
    return time.strftime("%I:%M:%S %p")


class Bridge(QObject):
    """Transporta eventos de audio desde los hilos de captura hasta Qt."""

    status_changed = pyqtSignal(str)
    subtitle_partial = pyqtSignal(str, str)
    subtitle_ready = pyqtSignal(str, str, int)
    subtitle_translated = pyqtSignal(str, str, int)


class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.bridge = Bridge()
        self.controller = None
        self._drag_pos = None
        self._last_segment_id = {
            "YOU": 0,
            "COMPANION": 0,
        }

        self._setup_window()
        self._setup_ui()
        self._connect_signals()
        self._schedule_resize()

    def _setup_window(self):
        # Se conserva el comportamiento actual: ventana flotante, movible
        # y siempre por encima de Teams, Zoom, navegador, IntelliJ, etc.
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumWidth(720)
        self.setMinimumHeight(330)
        self.resize(900, 420)
        self.move(60, 60)

    def _setup_ui(self):
        self.container = QFrame(self)
        self.container.setObjectName("mainCard")
        self.container.setStyleSheet("""
        QFrame#mainCard {
            background-color: rgba(15, 18, 22, 242);
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,20);
        }
        QLabel {
            color: white;
        }
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.container)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # Header
        title_bar = QHBoxLayout()
        title_bar.setSpacing(7)

        title_icon = QLabel("🎙")
        title_icon.setStyleSheet(
            "font-size: 14px; background: transparent;"
        )

        title = QLabel("Meeting Subtitles")
        title.setStyleSheet(
            "color: white; font-size: 15px; font-weight: bold; "
            "background: transparent;"
        )

        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(
            "color: #888888; font-size: 9px; background: transparent;"
        )

        self.status_label = QLabel("Inactivo")
        self.status_label.setStyleSheet(
            "color: #b8c4d0; font-size: 13px; background: transparent;"
        )

        title_bar.addWidget(title_icon)
        title_bar.addWidget(title)
        title_bar.addWidget(self.status_dot)
        title_bar.addWidget(self.status_label)
        title_bar.addStretch()

        self.always_on_top_label = QLabel("📌 Always on top")
        self.always_on_top_label.setStyleSheet(
            "color: #8f9baa; font-size: 11px; background: transparent;"
        )

        title_bar.addWidget(self.always_on_top_label)
        layout.addLayout(title_bar)

        # Controls
        controls = QHBoxLayout()
        controls.setSpacing(8)

        self.lang_combo = QComboBox()
        self.lang_combo.addItems([
            "🌐 English → Español",
            "🌐 Español → English",
        ])
        self.lang_combo.setCurrentIndex(0)
        self.lang_combo.setStyleSheet(PILL_STYLE)
        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)

        self.start_btn = QPushButton("▶ Iniciar")
        self.start_btn.setStyleSheet(START_BTN_STYLE)
        self.start_btn.clicked.connect(self._on_start_clicked)

        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(36, 36)
        self.close_btn.setStyleSheet(CLOSE_BTN_STYLE)
        self.close_btn.clicked.connect(self.close)

        controls.addWidget(self.lang_combo)
        controls.addStretch()
        controls.addWidget(self.start_btn)
        controls.addWidget(self.close_btn)
        layout.addLayout(controls)

        # Subtítulos
        subtitles = QHBoxLayout()
        subtitles.setSpacing(12)

        self.you_card = self._build_subtitle_card(
            "YOU",
            "#4da6ff",
            "rgba(30,45,70,205)",
        )
        self.companion_card = self._build_subtitle_card(
            "COMPANION",
            "#ffa64d",
            "rgba(65,45,25,205)",
        )

        subtitles.addWidget(self.you_card["card"])
        subtitles.addWidget(self.companion_card["card"])
        layout.addLayout(subtitles)

        # Estado de audio
        footer = QHBoxLayout()
        footer.setSpacing(12)

        self.mic_status = QLabel("🎤 Microphone: Ready")
        self.system_status = QLabel("🔊 System audio: Ready")

        for label in (self.mic_status, self.system_status):
            label.setStyleSheet(
                "color: #8f9baa; font-size: 11px; background: transparent;"
            )

        footer.addWidget(self.mic_status)
        footer.addWidget(self.system_status)
        footer.addStretch()
        layout.addLayout(footer)

        self._set_placeholders()

    def _build_subtitle_card(self, source, badge_color, background):
        card = QFrame()
        card.setStyleSheet(
            f"""
            QFrame {{
                background-color: {background};
                border-radius: 14px;
            }}
            """
        )

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 12, 14, 12)
        card_layout.setSpacing(8)

        header = QHBoxLayout()

        badge = QLabel(source)
        badge.setFixedHeight(22)
        badge.setStyleSheet(
            f"""
            background-color: {badge_color};
            color: white;
            font-size: 11px;
            font-weight: bold;
            border-radius: 11px;
            padding: 2px 12px;
            """
        )

        time_label = QLabel("")
        time_label.setStyleSheet(
            "color: #888888; font-size: 10px; background: transparent;"
        )
        time_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        header.addWidget(badge)
        header.addStretch()
        header.addWidget(time_label)
        card_layout.addLayout(header)

        english_label = QLabel("")
        english_label.setWordWrap(True)
        english_label.setMinimumHeight(70)
        english_label.setStyleSheet(
            "color: #e8e8e8; font-size: 14px; background: transparent;"
        )

        spanish_label = QLabel("")
        spanish_label.setWordWrap(True)
        spanish_label.setMinimumHeight(55)
        spanish_label.setStyleSheet(
            "color: #aeb9c5; font-size: 13px; background: transparent;"
        )

        card_layout.addWidget(english_label)
        card_layout.addWidget(spanish_label)
        card_layout.addStretch()

        return {
            "card": card,
            "time": time_label,
            "english": english_label,
            "spanish": spanish_label,
        }

    def _set_placeholders(self):
        for card in (self.you_card, self.companion_card):
            card["english"].setText("Waiting for speech...")
            card["english"].setStyleSheet(
                "color: #718092; font-size: 14px; font-style: italic; "
                "background: transparent;"
            )
            card["spanish"].setText("La traducción aparecerá aquí...")
            card["spanish"].setStyleSheet(
                "color: #718092; font-size: 13px; font-style: italic; "
                "background: transparent;"
            )
            card["time"].setText("")

    def _connect_signals(self):
        self.bridge.status_changed.connect(self._on_status_changed)
        self.bridge.subtitle_partial.connect(self._on_subtitle_partial)
        self.bridge.subtitle_ready.connect(self._on_subtitle_ready)
        self.bridge.subtitle_translated.connect(self._on_subtitle_translated)

    def _card_for(self, source):
        return self.you_card if source == "YOU" else self.companion_card

    def _on_language_changed(self, index):
        language = "en" if index == 0 else "es"
        set_meeting_language(language)

        if self.controller:
            self._restart_listener()

    def _restart_listener(self):
        if not self.controller:
            return

        old_controller = self.controller
        old_controller.stop()
        self.controller = None

        QTimer.singleShot(100, self._on_start_clicked)

    def _on_start_clicked(self):
        if self.controller:
            self._on_stop_clicked()
            return

        self.start_btn.setEnabled(False)
        self.start_btn.setText("Cargando...")

        callbacks = {
            "on_status": lambda text: self.bridge.status_changed.emit(text),
            "on_subtitle_partial": lambda source, text: (
                self.bridge.subtitle_partial.emit(source, text)
            ),
            "on_subtitle": lambda source, text, segment_id: (
                self.bridge.subtitle_ready.emit(source, text, segment_id)
            ),
            "on_subtitle_translated": lambda source, text, segment_id: (
                self.bridge.subtitle_translated.emit(source, text, segment_id)
            ),
        }

        self.controller = ListenerController(callbacks)

        threading.Thread(
            target=self.controller.start,
            name="subtitle-controller",
            daemon=True,
        ).start()

        self.start_btn.setText("■ Detener")
        self.start_btn.setEnabled(True)

    def _on_stop_clicked(self):
        if not self.controller:
            return

        self.controller.stop()
        self.controller = None
        self.start_btn.setText("▶ Iniciar")
        self._set_placeholders()
        self._schedule_resize()

    def _on_status_changed(self, text):
        listening = text.startswith("Escuchando")
        active = text not in ("Inactivo", "Detenido", "")

        color = (
            "#5cb85c"
            if listening
            else "#f0ad4e"
            if active
            else "#888888"
        )

        self.status_dot.setStyleSheet(
            f"color: {color}; font-size: 9px; background: transparent;"
        )
        self.status_label.setText(text or "Inactivo")

        if listening:
            self.mic_status.setText("🎤 Microphone: Connected")
            self.system_status.setText("🔊 System audio: Connected")
        elif text.startswith("Escuchando (solo YOU)"):
            self.mic_status.setText("🎤 Microphone: Connected")
            self.system_status.setText("🔊 System audio: Unavailable")
        else:
            self.mic_status.setText("🎤 Microphone: Ready")
            self.system_status.setText("🔊 System audio: Ready")

    def _on_subtitle_partial(self, source, text):
        card = self._card_for(source)
        card["english"].setStyleSheet(
            "color: #b8c4d0; font-size: 14px; font-style: italic; "
            "background: transparent;"
        )
        card["english"].setText(text)

    def _on_subtitle_ready(self, source, text, segment_id):
        if segment_id < self._last_segment_id[source]:
            return

        self._last_segment_id[source] = segment_id
        card = self._card_for(source)

        card["english"].setStyleSheet(
            "color: #e8e8e8; font-size: 14px; background: transparent;"
        )
        card["english"].setText(text)
        card["spanish"].setText("Traduciendo...")
        card["spanish"].setStyleSheet(
            "color: #718092; font-size: 13px; font-style: italic; "
            "background: transparent;"
        )
        card["time"].setText(_now())

    def _on_subtitle_translated(self, source, text, segment_id):
        if segment_id < self._last_segment_id[source]:
            return

        card = self._card_for(source)
        card["spanish"].setStyleSheet(
            "color: #d3dbe4; font-size: 13px; background: transparent;"
        )
        card["spanish"].setText(text)

    def _schedule_resize(self):
        QTimer.singleShot(0, self.adjustSize)

    # Ventana arrastrable, conservando el comportamiento de V1.
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = (
                event.globalPosition().toPoint()
                - self.frameGeometry().topLeft()
            )
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (
            self._drag_pos is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self.move(
                event.globalPosition().toPoint() - self._drag_pos
            )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def closeEvent(self, event):
        if self.controller:
            try:
                self.controller.stop()
            except Exception as exc:
                print(f"[UI] Error cerrando audio: {exc}")
            self.controller = None

        event.accept()


def main():
    app = QApplication(sys.argv)
    window = OverlayWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
