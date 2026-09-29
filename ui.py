import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faulthandler

faulthandler.enable()

import sys
import threading
import time
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QTextEdit, QFrame
)
from PyQt6.QtGui import QFont

from config import (
    set_language_mode, get_language_mode,
    set_assistant_enabled, get_assistant_enabled,
    set_assistant_listen_mode, get_assistant_listen_mode,
    get_ai_provider_display_name,
)
from vad_detector import ListenerController

SOURCE_COLORS = {
    "Tú": "#4da6ff",
    "Compañeros": "#ffa64d",
}

PILL_STYLE = """
    QPushButton, QComboBox {
        background-color: rgba(255,255,255,15);
        color: white;
        border-radius: 16px;
        padding: 6px 14px;
        border: 1px solid rgba(255,255,255,25);
    }
    QComboBox::drop-down { border: none; width: 18px; }
    QPushButton:hover, QComboBox:hover { background-color: rgba(255,255,255,25); }
    QComboBox QAbstractItemView {
        background-color: #1e1e1e;
        color: white;
        selection-background-color: #3a3a5a;
        selection-color: white;
        border: 1px solid rgba(255,255,255,30);
        outline: none;
    }
"""

ASSISTANT_BTN_ON_STYLE = """
    QPushButton {
        background-color: rgba(90,110,255,60);
        color: white;
        border-radius: 16px;
        padding: 6px 14px;
        border: 1px solid #5a6eff;
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

START_BTN_STYLE = """
    QPushButton {
        background-color: rgba(255,255,255,15);
        color: white;
        border-radius: 16px;
        padding: 6px 16px;
        border: 1px solid rgba(255,255,255,25);
        font-weight: bold;
    }
    QPushButton:hover { background-color: rgba(255,255,255,25); }
"""


def _now():
    return time.strftime("%I:%M %p")


class Bridge(QObject):
    """Reenvía callbacks del hilo de audio hacia la UI (thread-safe con señales Qt)."""
    status_changed = pyqtSignal(str)
    subtitle_ready = pyqtSignal(str, str, int)  # source, en_text, segment_id
    subtitle_translated = pyqtSignal(str, str, int)  # source, es_text, segment_id
    subtitle_partial = pyqtSignal(str, str)  # source, en_text (en construcción)
    question_ready = pyqtSignal(str, str)
    assistant_state_changed = pyqtSignal(str)
    answer_ready = pyqtSignal(str)
    answer_error = pyqtSignal(str)


class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.bridge = Bridge()
        self.controller = None
        self._drag_pos = None
        self._last_subtitle_segment_id = {"Tú": 0, "Compañeros": 0}
        self._is_listening = False

        self._setup_window()
        self._setup_ui()
        self._connect_signals()
        self._schedule_resize()

    # ---------- Ventana overlay ----------
    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumWidth(720)
        self.setMinimumHeight(390)
        self.move(60, 60)

    def _schedule_resize(self):
        QTimer.singleShot(0, self.adjustSize)

    def _setup_ui(self):
        self.container = QFrame(self)
        self.container.setObjectName("mainCard")
        self.container.setStyleSheet("""
            QFrame#mainCard {
                background-color: rgba(15, 18, 22, 240);
                border-radius: 16px;
                border: 1px solid rgba(255,255,255,20);
            }
            QLabel { color: white; }
        """)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.container)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        # --- Fila 1: Título (proveedor + estado unificado, tamaño compacto) ---
        title_bar = QHBoxLayout()
        title_bar.setSpacing(6)

        self.title_icon = QLabel("🐙")
        self.title_icon.setStyleSheet("font-size: 13px; background: transparent;")

        self.provider_label = QLabel(get_ai_provider_display_name())
        self.provider_label.setStyleSheet(
            "color: white; font-size: 14px; font-weight: bold; background: transparent;"
        )

        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #f0ad4e; font-size: 9px; background: transparent;")

        self.status_text_label = QLabel("Inactivo")
        self.status_text_label.setStyleSheet(
            "color: #b8c4d0; font-size: 13px; background: transparent;"
        )

        title_bar.addWidget(self.title_icon)
        title_bar.addWidget(self.provider_label)
        title_bar.addWidget(self.status_dot)
        title_bar.addWidget(self.status_text_label)
        title_bar.addStretch()
        layout.addLayout(title_bar)

        # --- Fila 2: Controles (pills) ---
        controls_bar = QHBoxLayout()
        controls_bar.setSpacing(8)

        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["🌐 Reunión: Español", "🌐 Reunión: Inglés"])
        self.lang_combo.setCurrentIndex(1)
        self.lang_combo.setStyleSheet(PILL_STYLE)
        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)

        self.assistant_btn = QPushButton("🤖 Asistente: OFF")
        self.assistant_btn.setCheckable(True)
        self.assistant_btn.setChecked(False)
        self.assistant_btn.setStyleSheet(PILL_STYLE)
        self.assistant_btn.clicked.connect(self._on_assistant_toggled)

        self.listen_mode_combo = QComboBox()
        self.listen_mode_combo.addItems([
            "👥 Responde a: Compañeros",
            "👥 Responde a: Yo",
            "👥 Responde a: Ambos",
        ])
        listen_mode_map = {"compañeros": 0, "yo": 1, "ambos": 2}
        self.listen_mode_combo.setCurrentIndex(listen_mode_map.get(get_assistant_listen_mode(), 2))
        self.listen_mode_combo.setStyleSheet(PILL_STYLE)
        self.listen_mode_combo.currentIndexChanged.connect(self._on_listen_mode_changed)

        self.start_btn = QPushButton("▶ Iniciar")
        self.start_btn.setStyleSheet(START_BTN_STYLE)
        self.start_btn.clicked.connect(self._on_start_clicked)

        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(36, 36)
        self.close_btn.setStyleSheet(CLOSE_BTN_STYLE)
        self.close_btn.clicked.connect(self.close)

        controls_bar.addWidget(self.lang_combo)
        controls_bar.addWidget(self.assistant_btn)
        controls_bar.addWidget(self.listen_mode_combo)
        controls_bar.addStretch()
        controls_bar.addWidget(self.start_btn)
        controls_bar.addWidget(self.close_btn)
        layout.addLayout(controls_bar)

        # --- Tarjetas de subtítulo: EN (inmediato) + ES (llega async después) ---
        subtitle_row = QHBoxLayout()
        subtitle_row.setSpacing(12)

        self.subtitle_card, self.subtitle_badge, self.subtitle_icon, self.subtitle_text, self.subtitle_time = \
            self._build_card("EN", "#4da6ff", "rgba(30,45,70,200)")
        self.subtitle_es_card, self.subtitle_es_badge, self.subtitle_es_icon, self.subtitle_es_text, self.subtitle_es_time = \
            self._build_card("ES", "#57c785", "rgba(20,45,35,200)")

        subtitle_row.addWidget(self.subtitle_card)
        subtitle_row.addWidget(self.subtitle_es_card)

        # --- Tarjetas de Pregunta/Respuesta (dependen solo del estado del asistente) ---
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)

        self.en_card, self.en_badge, self.en_icon, self.en_text, self.en_time = self._build_card(
            "PREGUNTA", "#4da6ff", "rgba(30,45,70,200)"
        )
        self.es_card, self.es_badge, self.es_icon, self.es_text, self.es_time = self._build_card(
            "RESPUESTA", "#57c785", "rgba(20,45,35,200)"
        )

        cards_row.addWidget(self.en_card)
        cards_row.addWidget(self.es_card)

        layout.addLayout(subtitle_row)
        layout.addLayout(cards_row)

        # --- Historial colapsable ---
        self.history_toggle_btn = QPushButton("▾ Historial")
        self.history_toggle_btn.setFlat(True)
        self.history_toggle_btn.setStyleSheet("color: #aaaaaa; text-align: left; border: none;")
        self.history_toggle_btn.clicked.connect(self._toggle_history)

        self.history_box = QTextEdit()
        self.history_box.setReadOnly(True)
        self.history_box.setVisible(False)
        self.history_box.setStyleSheet(
            "background-color: rgba(255,255,255,10); color: #dddddd; border: none; "
            "border-radius: 8px; font-size: 12px;"
        )
        self.history_box.setMaximumHeight(100)

        layout.addWidget(self.history_toggle_btn)
        layout.addWidget(self.history_box)

        self._set_placeholder_texts()
        self._apply_view_mode()

        self._on_language_changed(self.lang_combo.currentIndex())
        self._on_assistant_toggled(self.assistant_btn.isChecked())

    def _build_card(self, badge_text, badge_color, bg_color):
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border-radius: 14px;
            }}
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 12, 14, 12)
        card_layout.setSpacing(8)

        header = QHBoxLayout()
        badge = QLabel(badge_text)
        badge.setFixedHeight(22)
        badge.setStyleSheet(
            f"background-color: {badge_color}; color: white; font-size: 11px; "
            f"font-weight: bold; border-radius: 11px; padding: 2px 12px;"
        )
        header.addWidget(badge)
        header.addStretch()
        card_layout.addLayout(header)

        content_row = QHBoxLayout()
        content_row.setSpacing(8)
        icon = QLabel("💬")
        icon.setStyleSheet("font-size: 16px; background: transparent;")
        icon.setAlignment(Qt.AlignmentFlag.AlignTop)

        text_col = QVBoxLayout()
        text_col.setSpacing(4)
        text_label = QLabel("")
        text_label.setWordWrap(True)
        text_label.setStyleSheet("color: #d8d8d8; font-size: 14px; background: transparent;")

        time_label = QLabel("")
        time_label.setStyleSheet("color: #888888; font-size: 10px; background: transparent;")
        time_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        text_col.addWidget(text_label)
        text_col.addWidget(time_label)

        content_row.addWidget(icon)
        content_row.addLayout(text_col, stretch=1)
        card_layout.addLayout(content_row)
        card_layout.addStretch()

        return card, badge, icon, text_label, time_label

    def _set_placeholder_texts(self):
        self.subtitle_text.setText("Aquí aparecerá la transcripción en inglés...")
        self.subtitle_text.setStyleSheet("color: #7a8a9a; font-size: 14px; font-style: italic;")
        self.subtitle_time.setText("")

        self.subtitle_es_text.setText("La traducción aparecerá aquí...")
        self.subtitle_es_text.setStyleSheet("color: #7a8a9a; font-size: 14px; font-style: italic;")
        self.subtitle_es_time.setText("")

        self.en_text.setText("Esperando tu pregunta...")
        self.en_text.setStyleSheet("color: #7a8a9a; font-size: 14px; font-style: italic;")
        self.es_text.setText("La respuesta aparecerá aquí...")
        self.es_text.setStyleSheet("color: #7a8a9a; font-size: 14px; font-style: italic;")
        self.en_time.setText("")
        self.es_time.setText("")

    def _connect_signals(self):
        self.bridge.status_changed.connect(self._on_status_changed)
        self.bridge.subtitle_ready.connect(self._on_subtitle_ready)
        self.bridge.subtitle_translated.connect(self._on_subtitle_translated)
        self.bridge.subtitle_partial.connect(self._on_subtitle_partial)
        self.bridge.question_ready.connect(self._on_question_ready)
        self.bridge.assistant_state_changed.connect(self._on_assistant_state_changed)
        self.bridge.answer_ready.connect(self._on_answer_ready)
        self.bridge.answer_error.connect(self._on_answer_error)

    # ---------- Modo de vista: subtítulo único (independiente) + pregunta/respuesta (independiente) ----------
    def _apply_view_mode(self):
        assistant_on = get_assistant_enabled()
        is_english_meeting = get_language_mode() == "en"

        # Tarjetas de subtítulo: dependen SOLO del idioma de reunión (EN), sin importar el asistente
        self.subtitle_card.setVisible(is_english_meeting)
        self.subtitle_es_card.setVisible(is_english_meeting)

        # Tarjetas de pregunta/respuesta: dependen SOLO del estado del asistente
        self.en_card.setVisible(assistant_on)
        self.es_card.setVisible(assistant_on)

    def _apply_assistant_badges(self):
        lang_label = "EN" if get_language_mode() == "en" else "ES"
        self.en_badge.setText(f"PREGUNTA · {lang_label}")
        self.es_badge.setText(f"RESPUESTA · {lang_label}")

    # ---------- Handlers UI ----------
    def _on_language_changed(self, index):
        mode = "en" if index == 1 else "es"
        set_language_mode(mode)
        if get_assistant_enabled():
            self._apply_assistant_badges()
        self._apply_view_mode()

        # Si la app ya está corriendo, reinicia solo el micrófono con el nuevo idioma,
        # sin afectar el loopback de "Compañeros" ni pedir un "Detener" manual.
        if self.controller:
            threading.Thread(
                target=self.controller.restart_mic_language, args=(mode,), daemon=True
            ).start()

        self._schedule_resize()

    def _on_assistant_toggled(self, checked):
        set_assistant_enabled(checked)
        if checked:
            self.assistant_btn.setText("🤖 Asistente: ON")
            self.assistant_btn.setStyleSheet(ASSISTANT_BTN_ON_STYLE)
            self._apply_assistant_badges()
            self.en_icon.setText("🧑")
            self.es_icon.setText("🤖")
        else:
            self.assistant_btn.setText("🤖 Asistente: OFF")
            self.assistant_btn.setStyleSheet(PILL_STYLE)
            self.en_icon.setText("💬")
            self.es_icon.setText("💬")
        self._set_placeholder_texts()
        self._apply_view_mode()
        self._schedule_resize()

    def _toggle_history(self):
        visible = not self.history_box.isVisible()
        self.history_box.setVisible(visible)
        self.history_toggle_btn.setText("▴ Historial" if visible else "▾ Historial")
        self._schedule_resize()

    def _on_start_clicked(self):
        if self.controller:
            return
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Cargando...")

        callbacks = {
            "on_status": lambda text: self.bridge.status_changed.emit(text),
            "on_subtitle": lambda source, en, segment_id: self.bridge.subtitle_ready.emit(
                source, en, segment_id
            ),
            "on_subtitle_translated": lambda source, es, segment_id: self.bridge.subtitle_translated.emit(
                source, es, segment_id
            ),
            "on_subtitle_partial": lambda source, en: self.bridge.subtitle_partial.emit(source, en),
            "on_question": lambda source, q: self.bridge.question_ready.emit(source, q),
            "on_assistant_state": lambda text: self.bridge.assistant_state_changed.emit(text),
            "on_answer": lambda a: self.bridge.answer_ready.emit(a),
            "on_answer_error": lambda e: self.bridge.answer_error.emit(e),
        }

        self.controller = ListenerController(callbacks)
        threading.Thread(target=self.controller.start, daemon=True).start()
        self.start_btn.setText("■ Detener")
        self.start_btn.setEnabled(True)
        self.start_btn.clicked.disconnect()
        self.start_btn.clicked.connect(self._on_stop_clicked)

    def _on_stop_clicked(self):
        if self.controller:
            self.controller.stop()
            self.controller = None
        self.start_btn.setText("▶ Iniciar")
        self.start_btn.clicked.disconnect()
        self.start_btn.clicked.connect(self._on_start_clicked)

    # ---------- Slots conectados a señales ----------
    def _on_status_changed(self, text):
        self._is_listening = text.startswith("Escuchando")
        is_active = text not in ("Inactivo", "Detenido", "")

        color = "#5cb85c" if self._is_listening else ("#f0ad4e" if is_active else "#888888")
        self.status_dot.setStyleSheet(f"color: {color}; font-size: 9px; background: transparent;")
        self.status_text_label.setText(text if text else "Inactivo")
        self.provider_label.setText(get_ai_provider_display_name())
        self._schedule_resize()

    def _on_subtitle_partial(self, source, en_text):
        if get_language_mode() != "en":
            return

        color = SOURCE_COLORS.get(source, "#ffffff")
        self.subtitle_text.setStyleSheet("color: #b8c4d0; font-size: 14px; font-style: italic;")
        self.subtitle_text.setText(f'<span style="color:{color}; font-weight:bold;">[{source}]</span> {en_text}')
        self._schedule_resize()

    def _on_subtitle_ready(self, source, en_text, segment_id):
        if get_language_mode() != "en":
            return

        last_id = self._last_subtitle_segment_id.get(source, 0)
        if segment_id < last_id:
            return
        self._last_subtitle_segment_id[source] = segment_id

        color = SOURCE_COLORS.get(source, "#ffffff")
        self.subtitle_text.setStyleSheet("color: #e8e8e8; font-size: 14px;")
        self.subtitle_text.setText(f'<span style="color:{color}; font-weight:bold;">[{source}]</span> {en_text}')
        self.subtitle_time.setText(_now())
        self._append_history(f"[{source}] {en_text}")
        self._schedule_resize()

    def _on_subtitle_translated(self, source, es_text, segment_id):
        if get_language_mode() != "en":
            return

        last_id = self._last_subtitle_segment_id.get(source, 0)
        if segment_id < last_id:
            return  # llegó tarde, ya hay un segmento más nuevo mostrado

        color = SOURCE_COLORS.get(source, "#ffffff")
        self.subtitle_es_text.setStyleSheet("color: #e8e8e8; font-size: 14px;")
        self.subtitle_es_text.setText(f'<span style="color:{color}; font-weight:bold;">[{source}]</span> {es_text}')
        self.subtitle_es_time.setText(_now())
        self._schedule_resize()

    def _on_question_ready(self, source, question):
        if not get_assistant_enabled():
            return
        color = SOURCE_COLORS.get(source, "#ffd479")
        self.es_text.setStyleSheet("color: #7a8a9a; font-size: 14px; font-style: italic;")
        self.es_text.setText("Pensando...")
        self.es_time.setText("")

        self.en_text.setStyleSheet("color: #e8e8e8; font-size: 14px;")
        self.en_text.setText(f'<span style="color:{color}; font-weight:bold;">[{source}]</span> {question}')
        self.en_time.setText(_now())
        self._schedule_resize()

    def _on_assistant_state_changed(self, state_text):
        if not get_assistant_enabled():
            return
        self.es_text.setStyleSheet("color: #7a8a9a; font-size: 14px; font-style: italic;")
        self.es_text.setText(state_text)
        self._schedule_resize()

    def _on_answer_ready(self, answer):
        if not get_assistant_enabled():
            return
        self.es_text.setStyleSheet("color: #e8e8e8; font-size: 14px;")
        self.es_text.setText(answer)
        self.es_time.setText(_now())
        q = self.en_text.text()
        self._append_history(f"[P] {q}\n[R] {answer}")
        self._schedule_resize()

    def _on_answer_error(self, error_text):
        if not get_assistant_enabled():
            return
        self.es_text.setStyleSheet("color: #ff8888; font-size: 14px;")
        self.es_text.setText(error_text)
        self.es_time.setText(_now())
        self._schedule_resize()

    def _append_history(self, entry):
        self.history_box.append(entry)
        self.history_box.append("-" * 30)

    def _on_listen_mode_changed(self, index):
        mode_map = {0: "compañeros", 1: "yo", 2: "ambos"}
        set_assistant_listen_mode(mode_map.get(index, "ambos"))

    # ---------- Arrastrar ventana ----------
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def closeEvent(self, event):
        if self.controller:
            try:
                self.controller.stop()
            except Exception:
                pass
            self.controller = None
        event.accept()

        def _force_kill():
            time.sleep(4.0)
            os._exit(0)

        threading.Thread(target=_force_kill, daemon=True).start()


def main():
    app = QApplication(sys.argv)
    window = OverlayWindow()
    window.show()
    exit_code = app.exec()
    os._exit(exit_code)


if __name__ == "__main__":
    main()