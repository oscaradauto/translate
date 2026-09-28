import threading

LANGUAGE_MODE = "en"  # "en" o "es", se setea desde la UI
ASSISTANT_ENABLED = True  # True o False, se setea desde la UI
AI_PROVIDER = "github_models"  # "gemini" , "github_models", "ollama"


ASSISTANT_LISTEN_MODE = "ambos"

PROVIDER_DISPLAY_NAMES = {
    "gemini": "Gemini",
    "github_models": "GitHub Models",
    "ollama": "Ollama (Local)",
}

native_model_lock = threading.Lock()


def set_language_mode(mode):
    global LANGUAGE_MODE
    LANGUAGE_MODE = mode


def get_language_mode():
    return LANGUAGE_MODE


def set_assistant_enabled(enabled):
    global ASSISTANT_ENABLED
    ASSISTANT_ENABLED = enabled


def get_assistant_enabled():
    return ASSISTANT_ENABLED


def set_ai_provider(name):
    global AI_PROVIDER
    AI_PROVIDER = name


def get_ai_provider_name():
    return AI_PROVIDER


def get_ai_provider_display_name():
    return PROVIDER_DISPLAY_NAMES.get(AI_PROVIDER, AI_PROVIDER)


def set_assistant_listen_mode(mode):
    global ASSISTANT_LISTEN_MODE
    if mode not in ("compañeros", "yo", "ambos"):
        mode = "ambos"
    ASSISTANT_LISTEN_MODE = mode


def get_assistant_listen_mode():
    return ASSISTANT_LISTEN_MODE