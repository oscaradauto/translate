"""Conversation-turn assembly for Stage 2.

Whisper final segments are audio chunks, not necessarily complete interview
questions. This module keeps consecutive chunks from the same speaker together
until the controller decides that the assembled turn is answerable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionCandidate:
    assembly_id: int
    revision: int
    speaker: str
    text: str
    part_count: int


class QuestionAssembler:
    """Accumulate STT finals into a revisioned conversational turn."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._speaker = ""
        self._assembly_id = 0
        self._revision = 0
        self._next_assembly_id = 1

    def clear(self) -> None:
        self._parts.clear()
        self._speaker = ""
        self._assembly_id = 0
        self._revision = 0

    def has_pending(self, speaker: str | None = None) -> bool:
        if not self._parts:
            return False
        if speaker is None:
            return True
        return self._speaker == speaker

    def append(
        self,
        speaker: str,
        text: str,
    ) -> tuple[QuestionCandidate | None, QuestionCandidate]:
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            raise ValueError("QuestionAssembler cannot append empty text.")

        boundary_candidate = None
        if self._parts and self._speaker != speaker:
            boundary_candidate = self.snapshot()
            self._start_new(speaker, cleaned)
        elif not self._parts:
            self._start_new(speaker, cleaned)
        else:
            self._parts.append(cleaned)
            self._revision += 1

        current = self.snapshot()
        if current is None:
            raise RuntimeError("QuestionAssembler failed to create a candidate.")

        return boundary_candidate, current

    def snapshot(self) -> QuestionCandidate | None:
        if not self._parts or not self._speaker:
            return None

        return QuestionCandidate(
            assembly_id=self._assembly_id,
            revision=self._revision,
            speaker=self._speaker,
            text=" ".join(self._parts).strip(),
            part_count=len(self._parts),
        )

    def is_stale(self, candidate: QuestionCandidate) -> bool:
        """Return True only when the same active turn has a newer revision."""
        return (
            self._assembly_id == candidate.assembly_id
            and self._revision != candidate.revision
        )

    def claim(self, candidate: QuestionCandidate) -> bool:
        """Consume a current candidate before generating its answer.

        A candidate from an older, already-closed speaker boundary remains
        valid and does not touch the new active assembly.
        """
        if self._assembly_id != candidate.assembly_id:
            return True

        if self._revision != candidate.revision:
            return False

        self.clear()
        return True

    def discard(self, candidate: QuestionCandidate) -> None:
        """Discard only if the candidate is still the active exact revision."""
        if (
            self._assembly_id == candidate.assembly_id
            and self._revision == candidate.revision
        ):
            self.clear()

    def _start_new(self, speaker: str, text: str) -> None:
        self._parts = [text]
        self._speaker = speaker
        self._assembly_id = self._next_assembly_id
        self._next_assembly_id += 1
        self._revision = 1
