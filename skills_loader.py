"""Fast local retrieval for the Stage 2 senior-developer profile.

The YAML file is optional enrichment. If it is missing, invalid, or no skill
matches the current question, callers receive an empty context block and GPT
continues with its normal general knowledge + interview conversation context.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

import yaml

SKILLS_FILE_PATH = os.path.join(
    os.path.dirname(__file__),
    "knowledge",
    "skill.yaml",
)

_skills_cache: list[dict[str, Any]] | None = None
_profile_cache: dict[str, Any] | None = None
_last_mtime: float | None = None


def _normalize(text: str) -> str:
    folded = unicodedata.normalize("NFD", str(text).casefold())
    folded = "".join(
        char
        for char in folded
        if unicodedata.category(char) != "Mn"
    )
    return " ".join(
        re.sub(r"[^a-z0-9+#./ -]+", " ", folded).split()
    )


def _load_document() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    global _skills_cache, _profile_cache, _last_mtime

    if not os.path.exists(SKILLS_FILE_PATH):
        return {}, []

    try:
        mtime = os.path.getmtime(SKILLS_FILE_PATH)
        if (
            _skills_cache is not None
            and _profile_cache is not None
            and mtime == _last_mtime
        ):
            return _profile_cache, _skills_cache

        with open(SKILLS_FILE_PATH, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except (OSError, yaml.YAMLError) as exc:
        print(f"[SkillsLoader] No se pudo cargar skill.yaml: {exc}")
        return {}, []

    raw_profile = data.get("profile", {})
    raw_skills = data.get("skills", [])

    profile = raw_profile if isinstance(raw_profile, dict) else {}
    skills = [
        item
        for item in raw_skills
        if isinstance(item, dict)
    ] if isinstance(raw_skills, list) else []

    _profile_cache = profile
    _skills_cache = skills
    _last_mtime = mtime
    return profile, skills


def find_matching_skills(
    question_text: str,
    max_matches: int = 3,
) -> list[dict[str, Any]]:
    """Return the most relevant local profile entries without an API call."""
    normalized_question = _normalize(question_text)
    if not normalized_question:
        return []

    question_tokens = set(normalized_question.split())
    padded_question = f" {normalized_question} "

    def contains(phrase: str) -> bool:
        normalized_phrase = _normalize(phrase)
        if not normalized_phrase:
            return False

        phrase_tokens = normalized_phrase.split()
        if len(phrase_tokens) == 1:
            return phrase_tokens[0] in question_tokens

        return f" {normalized_phrase} " in padded_question

    _, skills = _load_document()
    scored: list[tuple[int, int, dict[str, Any]]] = []

    for index, skill in enumerate(skills):
        score = 0
        topic = str(skill.get("topic", ""))
        if topic and contains(topic):
            score += 8

        keywords = skill.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []

        for keyword in keywords:
            normalized_keyword = _normalize(keyword)
            if not normalized_keyword:
                continue
            if contains(normalized_keyword):
                # Longer phrases are more specific than one-word matches.
                score += 2 + min(5, len(normalized_keyword.split()))

        if score > 0:
            scored.append((score, -index, skill))

    scored.sort(
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )
    return [
        skill
        for _, _, skill in scored[: max(1, max_matches)]
    ]


def _language_value(
    item: dict[str, Any],
    field: str,
    language_mode: str,
) -> str:
    suffix = "es" if language_mode == "es" else "en"
    value = item.get(f"{field}_{suffix}", "")
    return " ".join(str(value).strip().split())


def build_context_packet(
    question_text: str,
    language_mode: str = "en",
    max_matches: int = 3,
) -> tuple[str, bool]:
    """Return compact profile context plus verified-experience availability.

    General senior-profile summaries are NOT evidence that the candidate has
    personally done something. First-person past experience is allowed only
    when a matched skill contains an explicit verified_experience_en/es field.
    """
    profile, _ = _load_document()
    matches = find_matching_skills(
        question_text,
        max_matches=max_matches,
    )
    if not matches:
        return "", False

    language_mode = "es" if language_mode == "es" else "en"
    lines: list[str] = []
    has_verified_experience = False

    profile_name = str(
        profile.get("name", "Senior Software Developer")
    ).strip()
    if profile_name:
        lines.append(f"Profile: {profile_name}")

    for skill in matches:
        topic = str(skill.get("topic", "")).strip()
        summary = _language_value(
            skill,
            "summary",
            language_mode,
        )
        guidance = _language_value(
            skill,
            "answer_guidance",
            language_mode,
        )
        verified_experience = _language_value(
            skill,
            "verified_experience",
            language_mode,
        )

        tools = skill.get("tools", [])
        if not isinstance(tools, list):
            tools = []
        tool_text = ", ".join(
            str(tool).strip()
            for tool in tools
            if str(tool).strip()
        )

        parts = [f"[{topic}]" if topic else "[Relevant skill]"]
        if summary:
            parts.append(summary)
        if tool_text:
            parts.append(f"Tools/technologies: {tool_text}.")
        if guidance:
            parts.append(f"Response guidance: {guidance}")
        if verified_experience:
            has_verified_experience = True
            parts.append(
                "Verified personal experience: "
                + verified_experience
            )

        lines.append(" ".join(parts))

    header = (
        "Relevant senior-developer profile context. General profile summaries "
        "describe strong senior-level knowledge and preferred approaches; they "
        "are NOT proof of personal past experience. Only text explicitly marked "
        "'Verified personal experience' may support claims such as 'I used', "
        "'I implemented', 'I configured', 'I led', or 'I handled'. Never invent "
        "employers, project names, dates, metrics, incidents, or achievements."
    )
    return header + "\n" + "\n".join(lines), has_verified_experience


def build_context_block(
    question_text: str,
    language_mode: str = "en",
    max_matches: int = 3,
) -> str:
    """Compatibility wrapper returning only the context text."""
    context, _ = build_context_packet(
        question_text,
        language_mode=language_mode,
        max_matches=max_matches,
    )
    return context
