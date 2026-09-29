import sys
import threading
import time

from PyQt6.QtCore import Qt, QObject, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


PILL_STYLE = """
QPushButton {
    background-color: rgba(255,255,255,15);
    color: white;
    border-radius: 16px;
    padding: 6px 14px;
    border: 1px solid rgba(255,255,255,25);
}
QPushButton:hover {
    background-color: rgba(255,255,255,25);
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
    shutdown_finished = pyqtSignal()


class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.bridge = Bridge()
        self.controller = None
        self._drag_pos = None
        self._entries = {}
        self._entry_order = []
        self._partial_entries = {}
        self._max_entries = 60
        self._starting = False
        self._closing = False
        self._shutdown_watchdog_armed = False

        self._history = []
        self._history_index = {}

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
        title_icon.setStyleSheet("font-size: 14px; background: transparent;")

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

        self.language_label = QLabel("🌐 English")
        self.language_label.setStyleSheet(
            "color: white; background-color: rgba(255,255,255,15); "
            "border-radius: 16px; padding: 6px 14px; "
            "border: 1px solid rgba(255,255,255,25);"
        )

        self.history_btn = QPushButton("🕘 Historial")
        self.history_btn.setStyleSheet(PILL_STYLE)
        self.history_btn.clicked.connect(self._show_history)

        self.start_btn = QPushButton("▶ Iniciar")
        self.start_btn.setStyleSheet(START_BTN_STYLE)
        self.start_btn.clicked.connect(self._on_start_clicked)

        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(36, 36)
        self.close_btn.setStyleSheet(CLOSE_BTN_STYLE)
        self.close_btn.clicked.connect(self.close)

        controls.addWidget(self.language_label)
        controls.addWidget(self.history_btn)
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
        self.bridge.shutdown_finished.connect(self._on_shutdown_finished)

    def _on_start_clicked(self):
        if self._starting or self._closing or self.controller:
            return

        self._starting = True
        self._clear_transcript()

        self.start_btn.setEnabled(False)
        self.start_btn.setText("⏳ Cargando...")
        self._on_status_changed("Cargando...")

        callbacks = {
            "on_status": lambda text: self.bridge.status_changed.emit(text),
            "on_subtitle_partial": lambda source, text: (
                self.bridge.subtitle_partial.emit(source, text)
            ),
            "on_subtitle": lambda source, text, segment_id: (
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

    def _clear_transcript(self):
        for entry in self._entries.values():
            entry["card"].deleteLater()

        self._entries.clear()
        self._entry_order.clear()
        self._partial_entries.clear()

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

        badge = QLabel(f"{SOURCE_LABELS.get(source, source)}-ENG")
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

        card_layout.addWidget(english)

        insert_at = max(0, self.transcript_layout.count() - 1)
        self.transcript_layout.insertWidget(insert_at, card)

        entry = {
            "card": card,
            "english": english,
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
        loading = text in ("Cargando...", "Preparando audio...")
        error = text.startswith("Error") or text == "API sin créditos"
        active = text not in ("Inactivo", "Detenido", "") and not error

        if listening:
            self._starting = False
            self.start_btn.setText("● En curso")
            self.start_btn.setEnabled(False)
        elif loading:
            self.start_btn.setText("⏳ Cargando...")
            self.start_btn.setEnabled(False)
        elif error:
            self._starting = False
            self.controller = None
            self.start_btn.setText("▶ Iniciar")
            self.start_btn.setEnabled(True)
        elif text == "Detenido":
            self._starting = False
            self.start_btn.setText("▶ Iniciar")
            self.start_btn.setEnabled(True)

        color = (
            "#5cb85c"
            if listening
            else "#d9534f"
            if error
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
        elif text.startswith("Escuchando (solo COMPANION)"):
            self.mic_status.setText("🎤 Microphone: Unavailable")
            self.system_status.setText("🔊 System audio: Connected")
        elif listening:
            self.mic_status.setText("🎤 Microphone: Connected")
            self.system_status.setText("🔊 System audio: Connected")
        else:
            self.mic_status.setText("🎤 Microphone: Ready")
            self.system_status.setText("🔊 System audio: Ready")

    def _on_subtitle_partial(self, source, text):
        entry = self._partial_entries.get(source)
        if entry is None:
            entry = self._ensure_entry(source, 0)
            self._partial_entries[source] = entry

        entry["english"].setText(text)

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

        history_key = self._entry_key(source, segment_id)
        history_item = {
            "key": history_key,
            "source": source,
            "time": _now(),
            "english": text,
        }
        self._history_index[history_key] = len(self._history)
        self._history.append(history_item)

        self._scroll_to_bottom()

    def _stop_controller_async(self, controller):
        def cleanup():
            try:
                controller.stop()
            except Exception as exc:
                print(f"[Shutdown] Error liberando recursos: {exc}")
            finally:
                # The cleanup is bounded by the controller/transport timeouts.
                # The Qt signal returns control to the UI thread so the
                # application can exit normally.
                self.bridge.shutdown_finished.emit()

        threading.Thread(
            target=cleanup,
            name="subtitle-shutdown",
            daemon=True,
        ).start()

    def _on_shutdown_finished(self):
        self.controller = None
        self._starting = False
        self._closing = False
        self._shutdown_watchdog_armed = False
        QApplication.quit()

    def _shutdown_watchdog_timeout(self):
        """Prevent a third-party cleanup stall from keeping the process alive."""
        if self._closing:
            self._closing = False
            QApplication.quit()

    def _show_history(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Historial de la conversación")
        dialog.setMinimumSize(720, 520)
        dialog.setModal(True)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #0f1216;
            }
            QLabel {
                background: transparent;
            }
        """)

        layout = QVBoxLayout(dialog)

        title = QLabel(
            f"Historial de la conversación · {len(self._history)} intervenciones"
        )
        title.setStyleSheet(
            "color: white; font-size: 15px; font-weight: bold;"
        )
        layout.addWidget(title)

        history_view = QTextBrowser()
        history_view.setOpenExternalLinks(False)
        history_view.setStyleSheet("""
            QTextBrowser {
                background-color: #0f1216;
                color: #e7ebef;
                border: 1px solid rgba(255,255,255,20);
                border-radius: 10px;
                padding: 10px;
                font-size: 13px;
            }
        """)

        if not self._history:
            history_view.setPlainText(
                "Todavía no hay intervenciones en el historial."
            )
        else:
            blocks = []
            for item in self._history:
                source = SOURCE_LABELS.get(item["source"], item["source"])
                english = item["english"]
                blocks.append(
                    f"[{item['time']}] {source}\n"
                    f"EN: {english}"
                )

            history_view.setPlainText("\n\n".join(blocks))

        layout.addWidget(history_view)

        buttons = QHBoxLayout()
        buttons.addStretch()

        copy_button = QPushButton("📋 Copiar todo")
        copy_button.setStyleSheet("""
            QPushButton {
                background-color: #242a32;
                color: #f1f4f7;
                border-radius: 16px;
                padding: 7px 14px;
                border: 1px solid #3a434d;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #2e3640;
            }
            QPushButton:disabled {
                color: #6f7b88;
                background-color: #1b2026;
                border-color: #2b323a;
            }
        """)
        copy_button.setEnabled(bool(self._history))

        def copy_history():
            clipboard = QApplication.clipboard()
            clipboard.setText(history_view.toPlainText())
            copy_button.setText("✓ Copiado")
            QTimer.singleShot(1200, lambda: copy_button.setText("📋 Copiar todo"))

        copy_button.clicked.connect(copy_history)
        buttons.addWidget(copy_button)

        close_button = QPushButton("Cerrar")
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #242a32;
                color: #f1f4f7;
                border-radius: 16px;
                padding: 7px 14px;
                border: 1px solid #3a434d;
            }
            QPushButton:hover {
                background-color: #2e3640;
            }
        """)
        close_button.clicked.connect(dialog.accept)
        buttons.addWidget(close_button)

        layout.addLayout(buttons)

        dialog.exec()

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
        if self._closing:
            event.accept()
            return

        if self.controller:
            self._closing = True
            self._starting = False
            self.start_btn.setEnabled(False)
            self.close_btn.setEnabled(False)
            self.history_btn.setEnabled(False)
            self.language_label.setEnabled(False)

            controller = self.controller
            self.controller = None

            # La ventana desaparece inmediatamente. El audio y los recursos del
            # motor de transcripción se liberan en segundo plano.
            self.hide()

            if not self._shutdown_watchdog_armed:
                self._shutdown_watchdog_armed = True
                QTimer.singleShot(8000, self._shutdown_watchdog_timeout)

            self._stop_controller_async(controller)
            event.ignore()
            return

        self._starting = False
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = OverlayWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
