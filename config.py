"""Configuración de Stage 1 - Meeting Subtitles.

Stage 1 se dedica exclusivamente a capturar y transcribir el audio de la
reunión en inglés. No se realiza traducción ni análisis de preguntas aquí.
"""

import os

MEETING_LANGUAGE = "en"

# Índice opcional del micrófono. Puede sobrescribirse con la variable de entorno
# MIC_DEVICE_INDEX para evitar depender de un índice fijo en otra máquina.
MIC_DEVICE_INDEX = int(os.getenv("MIC_DEVICE_INDEX", "1"))

# Modelo usado actualmente por RealtimeSTT. Stage 1 será migrado posteriormente
# a un motor especializado de streaming STT.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")

# Parámetros del motor temporal de RealtimeSTT durante la prueba de Stage 1.
REALTIME_COMPUTE_TYPE = os.getenv("REALTIME_COMPUTE_TYPE", "int8")
REALTIME_DEVICE = os.getenv("REALTIME_DEVICE", "cpu")


def get_meeting_language():
    return MEETING_LANGUAGE
