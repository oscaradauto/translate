from config import get_ai_provider_name
from ai_providers.gemini_provider import GeminiProvider
from ai_providers.github_models_provider import GitHubModelsProvider
from ai_providers.ollama_provider import OllamaProvider
from ai_providers.deepseek_provider import DeepSeekProvider


def get_provider():
    name = get_ai_provider_name()
    if name == "gemini":
        return GeminiProvider()
    elif name == "github_models":
        return GitHubModelsProvider()
    elif name == "ollama":
        return OllamaProvider()
    elif name == "deepseek":
        return DeepSeekProvider()
    return GeminiProvider()


def get_fallback_provider():
    """Proveedor de respaldo en la nube distinto al principal"""
    try:
        return GitHubModelsProvider()
    except Exception:
        return None