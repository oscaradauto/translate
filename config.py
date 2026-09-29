"""Configuración de la V2.0 - Meeting Subtitles.

La primera etapa de V2 está dedicada exclusivamente a subtítulos en reuniones.
No se carga ni se configura ningún agente de IA aquí.
"""

import os

MEETING_LANGUAGE = "en"
TRANSLATION_LANGUAGE = "es"

# Índice opcional del micrófono. Puede sobrescribirse con la variable de entorno
# MIC_DEVICE_INDEX para evitar depender de un índice fijo en otra máquina.
MIC_DEVICE_INDEX = int(os.getenv("MIC_DEVICE_INDEX", "1"))

# Modelo usado por Faster-Whisper para el audio de sistema.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")

# Modelo usado por RealtimeSTT para los subtítulos parciales del micrófono.
# En CPU, int8 evita que CTranslate2 intente cargar los pesos float16 y luego
# convertirlos automáticamente a float32.
REALTIME_COMPUTE_TYPE = os.getenv("REALTIME_COMPUTE_TYPE", "int8")
REALTIME_DEVICE = os.getenv("REALTIME_DEVICE", "cpu")

# El subtítulo en inglés debe aparecer sin esperar la traducción.
TRANSLATION_TIMEOUT = float(os.getenv("TRANSLATION_TIMEOUT", "4.0"))


def get_meeting_language():
    return MEETING_LANGUAGE


def set_meeting_language(language):
    global MEETING_LANGUAGE
    if language in ("en", "es"):
        MEETING_LANGUAGE = language
