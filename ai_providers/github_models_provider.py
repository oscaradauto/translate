import os
from openai import OpenAI
from dotenv import load_dotenv

from ai_providers.base import AIProvider

load_dotenv()

GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com"
REQUEST_TIMEOUT = 8  # segundos máximo antes de fallar y activar el fallback


class GitHubModelsProvider(AIProvider):
    def __init__(self, model_name="gpt-4o-mini"):
        token = os.environ.get("GITHUB_MODELS_TOKEN")
        self.client = OpenAI(
            base_url=GITHUB_MODELS_ENDPOINT,
            api_key=token,
            timeout=REQUEST_TIMEOUT,
            max_retries=0,
        )
        self.model_name = model_name

    def warmup(self):
        try:
            self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "hola"}],
                max_tokens=5,
            )
        except Exception as e:
            print(f"[GitHubModelsProvider] Warmup falló: {e}")

    def generate(self, system_prompt: str, prompt: str, temperature: float = 0.3,
                 max_tokens: int = 300) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[GitHubModelsProvider] Error: {e}")
            return ""