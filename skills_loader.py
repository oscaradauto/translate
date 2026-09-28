import os
import yaml

SKILLS_FILE_PATH = os.path.join(os.path.dirname(__file__), "knowledge", "skills.yaml")

_skills_cache = None
_last_mtime = None


def _load_raw_skills():
    global _skills_cache, _last_mtime
    if not os.path.exists(SKILLS_FILE_PATH):
        return []

    mtime = os.path.getmtime(SKILLS_FILE_PATH)
    if _skills_cache is not None and mtime == _last_mtime:
        return _skills_cache

    with open(SKILLS_FILE_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    _skills_cache = data.get("skills", [])
    _last_mtime = mtime
    return _skills_cache


def find_matching_skills(question_text, max_matches=2):
    """
    Busca qué entradas del YAML aplican a la pregunta, comparando keywords.
    Devuelve una lista de dicts de skill que coinciden.
    """
    skills = _load_raw_skills()
    text_lower = question_text.lower()

    matches = []
    for skill in skills:
        keywords = skill.get("keywords", [])
        if any(kw.lower() in text_lower for kw in keywords):
            matches.append(skill)
        if len(matches) >= max_matches:
            break

    return matches


def build_context_block(question_text, language_mode="en"):
    """
    Arma un bloque de texto con la info relevante del YAML para inyectar
    como contexto real en el prompt del asistente.
    """
    matches = find_matching_skills(question_text)
    if not matches:
        return ""

    lines = []
    for skill in matches:
        topic = skill.get("topic", "")
        years = skill.get("experience_years", "")
        is_strong = skill.get("is_strong_area", False)
        tools = skill.get("tools_used", [])

        if language_mode == "en":
            summary = skill.get("summary_en", "")
            example = skill.get("recent_production_example_en") or skill.get("recent_incident_en", "")
            strong_text = "Yes, this is one of my strongest areas." if is_strong else ""
        else:
            summary = skill.get("summary_es", "")
            example = skill.get("recent_production_example_es") or skill.get("recent_incident_es", "")
            strong_text = "Sí, es una de mis áreas más fuertes." if is_strong else ""

        block = f"[{topic}]"
        if years:
            block += f" Experiencia: {years}."
        if strong_text:
            block += f" {strong_text}"
        if summary:
            block += f" {summary}"
        if tools:
            block += f" Herramientas: {', '.join(tools)}."
        if example:
            block += f" Ejemplo reciente: {example}"

        lines.append(block)

    header = (
        "Real background information to use as ground truth when answering "
        "(do not just repeat it verbatim, phrase it naturally as your own answer):"
        if language_mode == "en"
        else
        "Información real de mi experiencia para usar como base al responder "
        "(no la repitas textual, formúlala de forma natural como tu propia respuesta):"
    )

    return header + "\n" + "\n".join(lines)