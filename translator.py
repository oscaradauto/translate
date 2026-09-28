import os
import deepl
from dotenv import load_dotenv

load_dotenv()
AUTH_KEY = os.environ.get("DEEPL_API_KEY")
translator = deepl.Translator(AUTH_KEY)


def translate_text(text, source="EN", target="ES"):
    if not text.strip():
        return ""
    try:
        result = translator.translate_text(text, source_lang=source, target_lang=target)
        return result.text
    except Exception as e:
        print(f"[Translator] Error: {e}")
        return ""