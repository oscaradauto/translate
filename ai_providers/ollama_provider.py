import requests
from ai_providers.base import AIProvider

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5-coder:1.5b"


class OllamaProvider(AIProvider):
    """
    Proveedor local usando Ollama (http://localhost:11434).
    Requiere tener Ollama corriendo (ya corre como servicio en Windows)
    y el modelo descargado (`ollama pull qwen2.5-coder:0.5b`).
    """

    def __init__(self, model_name=DEFAULT_MODEL):
        self.model_name = model_name

    def warmup(self):
        try:
            requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model_name,
                    "prompt": "hola",
                    "stream": False,
                    "keep_alive": "30m",
                },
                timeout=30,
            )
        except Exception as e:
            print(f"[OllamaProvider] Warmup falló: {e}")

    def generate(self, system_prompt: str, prompt: str, temperature: float = 0.3,
                 max_tokens: int = 200) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}"
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "num_ctx": 1024,
                    },
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as e:
            print(f"[OllamaProvider] Error: {e}")
            return ""