from config import get_ai_provider_name
from ai_providers.gemini_provider import GeminiProvider
from ai_providers.github_models_provider import GitHubModelsProvider
from ai_providers.ollama_provider import OllamaProvider


def get_provider():
    name = get_ai_provider_name()
    if name == "gemini":
        return GeminiProvider()
    elif name == "github_models":
        return GitHubModelsProvider()
    elif name == "ollama":
        return OllamaProvider()
    return GeminiProvider()


def get_fallback_provider():
    """Proveedor local de respaldo cuando el principal (nube) falla por conexión."""
    try:
        return OllamaProvider(model_name="qwen2.5-coder:1.5b")
    except Exception:
        return None