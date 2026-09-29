import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_ENDPOINT = "https://api.deepseek.com"
REQUEST_TIMEOUT = 6  # traducción corta, debe responder rápido

_client = OpenAI(
    base_url=DEEPSEEK_ENDPOINT,
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    timeout=REQUEST_TIMEOUT,
    max_retries=0,
)

TRANSLATE_SYSTEM_PROMPT = (
    "Eres un traductor. Traduce el siguiente texto de inglés a español. "
    "Responde SOLO con la traducción, sin explicaciones, sin comillas, sin notas adicionales."
)

def translate_text(text, source="EN", target="ES"):
    """
    Traduce texto usando DeepSeek (rápido y económico).
    Mantiene la misma firma que la versión anterior basada en DeepL
    para no romper el resto del código que la invoca.
    """
    if not text or not text.strip():
        return ""
    try:
        response = _client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
         print(f"[Translator] Error: {e}")
         return ""