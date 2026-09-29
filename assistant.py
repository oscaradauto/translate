import re
import threading
import unicodedata

from config import get_language_mode
from ai_providers import get_provider, get_fallback_provider
from skills_loader import build_context_block

SYSTEM_PROMPT_ES = (
    "Estás en una entrevista técnica de trabajo como desarrollador senior, respondiendo de palabra en tiempo real. "
    "Responde SIEMPRE en español, nunca mezcles inglés. "
    "PROHIBIDO iniciar con frases como 'Buena pregunta', 'Claro', 'Sí, puedo responder' o cualquier confirmación. "
    "PROHIBIDO usar negritas, asteriscos, numeración, viñetas o frases como 'aquí te explico' o 'aquí tienes'. "
    "PROHIBIDO estructurar la respuesta en partes o elementos, aunque la pregunta mencione varios conceptos. "
    "Si la pregunta menciona varios conceptos (ej: CI y CD, JWT y OAuth), fusiona la idea en UNA sola frase comparativa. "
    "Si no estás completamente seguro de un dato específico (número exacto, versión, estadística), "
    "no inventes cifras: responde con el concepto general en vez de un número inventado. "
    "Si se te proporciona información real de experiencia (background), úsala como base de tu respuesta, "
    "pero exprésala de forma natural, no la copies literalmente. "
    "Estás en el contexto EXCLUSIVO de una entrevista de trabajo de desarrollo de software. "
    "Si una palabra de la pregunta no tiene sentido en un contexto técnico de programación "
    "(ej. nombres propios, lugares, o palabras que no son términos de software), asume que es un error "
    "de transcripción de audio por voz y responde sobre el término técnico más parecido fonéticamente "
    "(ej. 'Chava' probablemente significa 'Java', 'Indicap' probablemente significa 'GitHub'), "
    "sin mencionar el error ni aclarar la corrección, solo responde directamente sobre el tema técnico correcto. "
    "Habla como una persona real conversando, NO como un texto escrito o un ensayo. "
    "Usa conectores naturales de alguien hablando, como 'básicamente', 'o sea', 'en el fondo', 'digamos que', "
    "'la idea es que', 'al final', 'lo que pasa es que', en vez de conectores formales como 'mientras que', "
    "'no obstante', 'por otro lado', 'cabe destacar'. "
    "Usa frases cortas y directas, como si lo dijeras de corrido sin pensar mucho la estructura, evitando subordinadas largas. "
    "Responde con seguridad y precisión técnica, como si lo estuvieras explicando al entrevistador para demostrar que dominas el tema. "
    "Si aplica, menciona brevemente un caso de uso o ejemplo real para reforzar la respuesta, sin extenderte. "
    "Ve directo a la respuesta desde la primera palabra. "
    "Da SOLO 1 idea central en 2 o 3 frases cortas, como si lo dijeras hablando en la entrevista, sin ningún tipo de formato."
)

SYSTEM_PROMPT_EN = (
    "You are in a technical job interview as a senior developer, answering out loud in real time. "
    "Respond ALWAYS in English, never mix Spanish. "
    "FORBIDDEN to start with phrases like 'Great question', 'Sure', 'Sure, I can answer', or any confirmation. "
    "FORBIDDEN to use bold text, asterisks, numbered items, bullet points, or phrases like 'here is' or 'let me explain'. "
    "FORBIDDEN to structure the answer into parts or elements, even if the question mentions several concepts. "
    "If the question mentions several concepts (e.g. CI and CD, JWT and OAuth), merge the idea into ONE comparative sentence. "
    "If you are not completely sure of a specific fact (exact number, version, statistic), "
    "do not invent figures: answer with the general concept instead of a made-up number. "
    "If real background information is provided, use it as the basis of your answer, "
    "but phrase it naturally, do not copy it verbatim. "
    "You are in the EXCLUSIVE context of a software development job interview. "
    "If a word in the question does not make sense in a technical programming context "
    "(e.g. proper names, places, or words that are not software terms), assume it is a voice "
    "transcription error and answer about the closest phonetically matching technical term "
    "(e.g. 'Chava' likely means 'Java', 'Indicap' likely means 'GitHub'), "
    "without mentioning the error or clarifying the correction, just answer directly about the correct technical topic. "
    "Talk like a real person having a conversation, NOT like written text or an essay. "
    "Use natural spoken connectors, like 'basically', 'so', 'the thing is', 'let's say', "
    "'at the end of the day', 'what happens is', instead of formal connectors like 'however', "
    "'nevertheless', 'on the other hand', 'it is worth noting'. "
    "Use short, direct sentences, as if you were saying it off the top of your head, avoiding long subordinate clauses. "
    "Answer with confidence and technical precision, as if explaining to the interviewer to demonstrate you master the topic. "
    "If relevant, briefly mention a real use case or example to reinforce the answer, without going too long. "
    "Go straight to the answer from the first word. "
    "Give ONLY 1 core idea in 2 or 3 short sentences, like you're saying it out loud in the interview, with no formatting at all."
)


def _strip_accents(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )


QUESTION_STARTERS = (
    "what", "how", "why", "when", "where", "who", "which", "whats",
    "can you", "could you", "do you", "does anyone", "would you",
    "is it", "are there", "should we", "is there",
    "que", "como", "por que", "para que", "cuando", "donde", "quien",
    "cual", "cuales", "cuanto", "cuantos", "cuantas",
    "puedes", "podrias", "sabes",
)

REQUEST_PHRASES = (
    "cuentame", "cuentame acerca de", "cuentame sobre",
    "explicame", "explicame acerca de", "explicame sobre",
    "dime", "dime acerca de", "hablame de", "hablame sobre",
    "hablame acerca de",
    "quiero saber", "me gustaria saber", "no se que es", "no se como",
    "me puedes explicar", "me podrias explicar", "me puedes decir",
    "ayudame a entender",
    "tell me about", "tell me", "explain", "explain to me",
    "walk me through", "i want to know", "help me understand",
    "can you explain", "could you explain",
)

QUESTION_FRAGMENTS = (
    "que es", "que son", "que significa",
    "para que sirve", "para que sirven",
    "como funciona", "como funcionan", "como se usa",
    "cual es la diferencia", "cuales son las diferencias",
    "diferencia entre",
    "what is", "what are", "what does",
    "how does", "how do", "how to",
    "difference between",
    "como configuro", "como configuras", "how do i configure",
    "que pipeline", "cual pipeline",
)


def is_question(text):
    original = text.strip()
    text = _strip_accents(original.lower())
    text = re.sub(r'\bhabla\s+me\b', 'hablame', text)
    text = re.sub(r'\bcuenta\s+me\b', 'cuentame', text)
    text = re.sub(r'\bexplica\s+me\b', 'explicame', text)

    if original.endswith("?"):
        return True
    if text.startswith(QUESTION_STARTERS):
        return True
    if text.startswith(REQUEST_PHRASES):
        return True
    if any(phrase in text for phrase in REQUEST_PHRASES):
        return True
    if any(phrase in text for phrase in QUESTION_FRAGMENTS):
        return True
    return False


def _truncate_to_max_sentences(text, max_sentences=3):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    complete_sentences = [s for s in sentences if s.strip().endswith((".", "!", "?"))]

    if not complete_sentences:
        # Si no hay ni una oración completa (respuesta cortada muy pronto),
        # se muestra el texto tal cual en vez de nada.
        return text.strip()

    complete_sentences = complete_sentences[:max_sentences]
    return " ".join(complete_sentences).strip()


conversation_history = []
MAX_HISTORY_TURNS = 3

_provider = None
_provider_lock = threading.Lock()


def _get_active_provider():
    global _provider
    from config import get_ai_provider_name

    with _provider_lock:
        if _provider is None or getattr(_provider, "_provider_name", None) != get_ai_provider_name():
            _provider = get_provider()
            _provider._provider_name = get_ai_provider_name()
        return _provider


def warmup():
    _get_active_provider().warmup()


def answer_question(question):
    mode = get_language_mode()
    system_prompt = SYSTEM_PROMPT_EN if mode == "en" else SYSTEM_PROMPT_ES

    history_text = ""
    for turn in conversation_history[-MAX_HISTORY_TURNS * 2:]:
        role = "Usuario" if turn["role"] == "user" else "Asistente"
        history_text += f"{role}: {turn['content']}\n"

    context_block = build_context_block(question, language_mode=mode)
    context_section = f"{context_block}\n\n" if context_block else ""

    prompt = f"{context_section}{history_text}Usuario: {question}\nAsistente:"

    text = _get_active_provider().generate(system_prompt, prompt, temperature=0.3, max_tokens=450)

    if not text:
        # Fallback automático a DeepSeek si el proveedor principal falla
        fallback = get_fallback_provider()
        if fallback:
            print("[Assistant] Proveedor principal falló, usando fallback en la nube (DeepSeek)...")
            text = fallback.generate(system_prompt, prompt, temperature=0.3, max_tokens=200)

    if not text:
        return None

    answer = _truncate_to_max_sentences(text, max_sentences=3)

    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": answer})

    # Recorta la lista para no crecer indefinidamente en sesiones largas
    max_items = MAX_HISTORY_TURNS * 2
    if len(conversation_history) > max_items:
        del conversation_history[:-max_items]
    return answer