import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import threading
import time

from PyQt6.QtCore import Qt, QObject, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from config import set_meeting_language


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

SOURCE_LABELS = {
    "YOU": "TU",
    "COMPANION": "COMPAÑERO",
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
        self._entries = {}
        self._entry_order = []
        self._max_entries = 60

        self._setup_window()
        self._setup_ui()
        self._connect_signals()

    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumWidth(720)
        self.setMinimumHeight(380)
        self.resize(900, 520)
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

        controls = QHBoxLayout()
        controls.setSpacing(8)

        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["🌐 English → Español"])
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

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setStyleSheet("""
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
        """)

        self.transcript_widget = QWidget()
        self.transcript_widget.setStyleSheet("background: transparent;")
        self.transcript_layout = QVBoxLayout(self.transcript_widget)
        self.transcript_layout.setContentsMargins(2, 2, 2, 2)
        self.transcript_layout.setSpacing(8)
        self.transcript_layout.addStretch()

        self.scroll_area.setWidget(self.transcript_widget)
        layout.addWidget(self.scroll_area, 1)

        self.empty_label = QLabel(
            "Inicia la reunión para ver los subtítulos aquí."
        )
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet(
            "color: #718092; font-size: 13px; background: transparent;"
        )
        self.transcript_layout.insertWidget(0, self.empty_label)

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

    def _connect_signals(self):
        self.bridge.status_changed.connect(self._on_status_changed)
        self.bridge.subtitle_partial.connect(self._on_subtitle_partial)
        self.bridge.subtitle_ready.connect(self._on_subtitle_ready)
        self.bridge.subtitle_translated.connect(self._on_subtitle_translated)

    def _on_language_changed(self, index):
        set_meeting_language("en")
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

        self._clear_transcript()
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

        # Importamos RealtimeSTT/soundcard después de crear QApplication.
        # Algunos módulos de audio inicializan COM en Windows; hacerlo antes
        # de Qt puede provocar el error OleInitialize()/0x80010106.
        from vad_detector import ListenerController

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
        self._clear_transcript()

    def _clear_transcript(self):
        for entry in self._entries.values():
            entry["card"].deleteLater()

        self._entries.clear()
        self._entry_order.clear()

        if self.empty_label is None:
            self.empty_label = QLabel(
                "Inicia la reunión para ver los subtítulos aquí."
            )
            self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.empty_label.setStyleSheet(
                "color: #718092; font-size: 13px; background: transparent;"
            )
            self.transcript_layout.insertWidget(0, self.empty_label)
        else:
            self.empty_label.show()

    def _entry_key(self, source, segment_id):
        return source, segment_id

    def _ensure_entry(self, source, segment_id):
        key = self._entry_key(source, segment_id)
        if key in self._entries:
            return self._entries[key]

        if self.empty_label:
            self.empty_label.hide()

        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: rgba(255,255,255,8);
                border-radius: 12px;
                border: 1px solid rgba(255,255,255,15);
            }
        """)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 9, 12, 10)
        card_layout.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(7)

        badge = QLabel(
            f"{SOURCE_LABELS.get(source, source)}-ENG"
        )
        badge.setStyleSheet(
            f"background-color: {SOURCE_COLORS.get(source, '#888888')}; "
            "color: white; font-size: 10px; font-weight: bold; "
            "border-radius: 9px; padding: 3px 8px;"
        )

        time_label = QLabel(_now())
        time_label.setStyleSheet(
            "color: #687584; font-size: 10px; background: transparent;"
        )

        header.addWidget(badge)
        header.addStretch()
        header.addWidget(time_label)
        card_layout.addLayout(header)

        english = QLabel("")
        english.setWordWrap(True)
        english.setStyleSheet(
            "color: #f0f2f5; font-size: 14px; background: transparent;"
        )

        spanish_badge = QLabel(f"{SOURCE_LABELS.get(source, source)}-ES")
        spanish_badge.setStyleSheet(
            "color: #9aa9b8; font-size: 10px; font-weight: bold; "
            "background: transparent;"
        )

        spanish = QLabel("Traduciendo...")
        spanish.setWordWrap(True)
        spanish.setStyleSheet(
            "color: #718092; font-size: 13px; font-style: italic; "
            "background: transparent;"
        )

        card_layout.addWidget(english)
        card_layout.addWidget(spanish_badge)
        card_layout.addWidget(spanish)

        insert_at = max(0, self.transcript_layout.count() - 1)
        self.transcript_layout.insertWidget(insert_at, card)

        entry = {
            "card": card,
            "english": english,
            "spanish": spanish,
        }
        self._entries[key] = entry
        self._entry_order.append(key)

        while len(self._entry_order) > self._max_entries:
            old_key = self._entry_order.pop(0)
            old_entry = self._entries.pop(old_key, None)
            if old_entry:
                old_entry["card"].deleteLater()

        return entry

    def _scroll_to_bottom(self):
        QTimer.singleShot(
            0,
            lambda: self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            ),
        )

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

        if text.startswith("Escuchando (solo YOU)"):
            self.mic_status.setText("🎤 Microphone: Connected")
            self.system_status.setText("🔊 System audio: Unavailable")
        elif listening:
            self.mic_status.setText("🎤 Microphone: Connected")
            self.system_status.setText("🔊 System audio: Connected")
        else:
            self.mic_status.setText("🎤 Microphone: Ready")
            self.system_status.setText("🔊 System audio: Ready")

    def _on_subtitle_partial(self, source, text):
        # El parcial se muestra inmediatamente en el mismo bloque de esa voz.
        entry = self._partial_entries.get(source)
        if entry is None:
            entry = self._ensure_entry(source, 0)
            self._partial_entries[source] = entry

        entry["english"].setText(text)
        entry["spanish"].setText("")

        self._scroll_to_bottom()

    def _on_subtitle_ready(self, source, text, segment_id):
        partial_key = (source, 0)
        partial_entry = self._partial_entries.pop(source, None)

        if partial_entry is not None:
            self._entries.pop(partial_key, None)
            entry = partial_entry
            actual_key = self._entry_key(source, segment_id)
            self._entries[actual_key] = entry
            if partial_key in self._entry_order:
                self._entry_order[self._entry_order.index(partial_key)] = actual_key
        else:
            entry = self._ensure_entry(source, segment_id)

        entry["english"].setText(text)
        entry["spanish"].setText("Traduciendo...")
        entry["spanish"].setStyleSheet(
            "color: #718092; font-size: 13px; font-style: italic; "
            "background: transparent;"
        )
        self._scroll_to_bottom()

    def _on_subtitle_translated(self, source, text, segment_id):
        entry = self._ensure_entry(source, segment_id)
        entry["spanish"].setStyleSheet(
            "color: #d3dbe4; font-size: 13px; background: transparent;"
        )
        entry["spanish"].setText(text)
        self._scroll_to_bottom()

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
