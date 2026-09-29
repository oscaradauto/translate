import os
from openai import OpenAI
from dotenv import load_dotenv

from ai_providers.base import AIProvider

load_dotenv()

DEEPSEEK_ENDPOINT = "https://api.deepseek.com"
REQUEST_TIMEOUT = 10


class DeepSeekProvider(AIProvider):
    def __init__(self, model_name="deepseek-chat"):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.client = OpenAI(
            base_url=DEEPSEEK_ENDPOINT,
            api_key=api_key,
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
            print(f"[DeepSeekProvider] Warmup falló: {e}")

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
            print(f"[DeepSeekProvider] Error: {e}")
            return ""