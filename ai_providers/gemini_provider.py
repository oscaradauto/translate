import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

from ai_providers.base import AIProvider

load_dotenv()


class GeminiProvider(AIProvider):
    def __init__(self, model_name="gemini-3.8-flash"):
        api_key = os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def warmup(self):
        try:
            self.client.models.generate_content(model=self.model_name, contents="hola")
        except Exception as e:
            print(f"[GeminiProvider] Warmup falló: {e}")

    def generate(self, system_prompt: str, prompt: str, temperature: float = 0.3,
                 max_tokens: int = 500) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}"
        max_retries = 2
        backoff_seconds = 1.5

        for attempt in range(max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )

                finish_reason = None
                if response.candidates:
                    finish_reason = getattr(response.candidates[0], "finish_reason", None)

                text = (response.text or "").strip()

                if finish_reason is not None and str(finish_reason) == "MAX_TOKENS" and attempt < max_retries:
                    print(f"[GeminiProvider] Respuesta cortada por límite de tokens, reintentando con más tokens...")
                    max_tokens = int(max_tokens * 1.5)
                    continue

                return text

            except Exception as e:
                error_text = str(e)
                is_overloaded = "503" in error_text or "UNAVAILABLE" in error_text

                if is_overloaded and attempt < max_retries:
                    print(f"[GeminiProvider] Sobrecargado, reintentando ({attempt + 1}/{max_retries})...")
                    time.sleep(backoff_seconds)
                    backoff_seconds *= 2
                    continue

                print(f"[GeminiProvider] Error: {e}")
                return ""

        return ""