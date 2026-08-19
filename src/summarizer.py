"""
summarizer.py
-------------
Turns a raw transcript into:
  - a coherent overview summary (map-reduce over BART)
  - a list of key points / decisions
  - a list of structured action items (task / owner / due date)

Two backends are supported:
  1. Local, fully offline: facebook/bart-large-cnn for summarization +
     google/flan-t5-base for structured extraction.
  2. OpenAI: a single GPT call that returns structured JSON directly.
     Used automatically when OPENAI_API_KEY is set and use_openai=True.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

from .utils import chunk_text


@dataclass
class ActionItem:
    task: str
    owner: Optional[str] = None
    due: Optional[str] = None


@dataclass
class SummaryResult:
    summary: str
    key_points: List[str]
    action_items: List[ActionItem]


# --------------------------------------------------------------------------
# Local (Hugging Face) backend
# --------------------------------------------------------------------------

class LocalSummarizer:
    """Fully offline summarizer using BART (summary) + FLAN-T5 (extraction)."""

    def __init__(
        self,
        summarizer_model: str = "facebook/bart-large-cnn",
        extractor_model: str = "google/flan-t5-base",
        device: Optional[int] = None,
    ):
        # Imported lazily so `import src.summarizer` stays cheap until used.
        from transformers import pipeline
        import torch

        resolved_device = device
        if resolved_device is None:
            resolved_device = 0 if torch.cuda.is_available() else -1

        self._bart = pipeline(
            "summarization", model=summarizer_model, device=resolved_device
        )
        self._flan = pipeline(
            "text2text-generation", model=extractor_model, device=resolved_device
        )

    # ---- summary (map-reduce) -------------------------------------------

    def _summarize_chunk(self, chunk: str) -> str:
        word_count = len(chunk.split())
        max_len = max(30, min(180, word_count // 2))
        min_len = max(15, max_len // 3)
        result = self._bart(
            chunk, max_length=max_len, min_length=min_len, do_sample=False
        )
        return result[0]["summary_text"].strip()

    def summarize(self, transcript: str, chunk_words: int = 1000) -> str:
        chunks = chunk_text(transcript, max_words=chunk_words)
        chunk_summaries = [self._summarize_chunk(c) for c in chunks]

        if len(chunk_summaries) == 1:
            return chunk_summaries[0]

        # Reduce step: summarize the summaries into one coherent overview.
        combined = " ".join(chunk_summaries)
        return self._summarize_chunk(combined)

    # ---- key points --------------------------------------------------

    def extract_key_points(self, transcript: str, chunk_words: int = 1000) -> List[str]:
        chunks = chunk_text(transcript, max_words=chunk_words)
        points: List[str] = []
        for chunk in chunks[:6]:  # cap work for very long transcripts
            prompt = (
                "List the key points and decisions made in this meeting "
                "excerpt as short bullet sentences, one per line, no numbering:\n\n"
                f"{chunk}"
            )
            output = self._flan(prompt, max_length=200, do_sample=False)[0][
                "generated_text"
            ]
            points.extend(self._split_lines(output))
        return self._dedupe(points)[:10]

    # ---- action items ---------------------------------------------------

    def extract_action_items(
        self, transcript: str, chunk_words: int = 1000
    ) -> List[ActionItem]:
        chunks = chunk_text(transcript, max_words=chunk_words)
        raw_items: List[str] = []
        for chunk in chunks[:6]:
            prompt = (
                "Extract concrete action items / tasks / to-dos from this "
                "meeting excerpt. For each, output one line in the exact "
                "format: Task | Owner | Due. Use 'None' if owner or due "
                "date isn't mentioned. If there are no action items, "
                "output nothing.\n\n"
                f"{chunk}"
            )
            output = self._flan(prompt, max_length=200, do_sample=False)[0][
                "generated_text"
            ]
            raw_items.extend(self._split_lines(output))

        items = [self._parse_action_line(line) for line in raw_items]
        items = [i for i in items if i is not None]
        return self._dedupe_items(items)[:15]

    # ---- helpers ----------------------------------------------------

    @staticmethod
    def _split_lines(text: str) -> List[str]:
        lines = re.split(r"[\n;]|(?<=\.)\s(?=[A-Z])", text)
        cleaned = [re.sub(r"^[\-\*\d\.\)\s]+", "", l).strip() for l in lines]
        return [l for l in cleaned if l and len(l) > 3]

    @staticmethod
    def _dedupe(items: List[str]) -> List[str]:
        seen = set()
        result = []
        for item in items:
            key = item.lower().strip()
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    @staticmethod
    def _dedupe_items(items: List[ActionItem]) -> List[ActionItem]:
        seen = set()
        result = []
        for item in items:
            key = item.task.lower().strip()
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    @staticmethod
    def _parse_action_line(line: str) -> Optional[ActionItem]:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3:
            task, owner, due = parts
        elif len(parts) == 1:
            task, owner, due = parts[0], "None", "None"
        else:
            task, owner, due = parts[0], (parts[1] if len(parts) > 1 else "None"), "None"

        if not task or task.lower() in {"none", "n/a", "-"}:
            return None

        return ActionItem(
            task=task,
            owner=None if owner.lower() in {"none", "n/a", "-", ""} else owner,
            due=None if due.lower() in {"none", "n/a", "-", ""} else due,
        )

    def run(self, transcript: str, chunk_words: int = 1000) -> SummaryResult:
        return SummaryResult(
            summary=self.summarize(transcript, chunk_words),
            key_points=self.extract_key_points(transcript, chunk_words),
            action_items=self.extract_action_items(transcript, chunk_words),
        )


# --------------------------------------------------------------------------
# OpenAI backend
# --------------------------------------------------------------------------

class OpenAISummarizer:
    """Single-call structured summarization using an OpenAI chat model."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        from openai import OpenAI

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def run(self, transcript: str, chunk_words: int = 1000) -> SummaryResult:
        # Long transcripts: summarize in chunks first, then do one final
        # structured pass over the concatenated chunk summaries.
        chunks = chunk_text(transcript, max_words=chunk_words)
        if len(chunks) > 1:
            partials = [self._quick_summarize(c) for c in chunks]
            source_text = "\n\n".join(partials)
        else:
            source_text = transcript

        prompt = f"""You are an assistant that turns meeting/lecture transcripts
into structured notes. Given the transcript (or transcript summary) below,
return ONLY valid JSON with this exact shape, no markdown fences, no commentary:

{{
  "summary": "a concise 3-6 sentence overview of what was discussed",
  "key_points": ["short bullet point", "..."],
  "action_items": [
    {{"task": "concrete task", "owner": "name or null", "due": "date/deadline or null"}}
  ]
}}

Transcript:
{source_text}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = response.choices[0].message.content.strip()
        content = re.sub(r"^```(json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Fall back gracefully if the model didn't return clean JSON.
            return SummaryResult(summary=content, key_points=[], action_items=[])

        action_items = [
            ActionItem(
                task=a.get("task", "").strip(),
                owner=a.get("owner") or None,
                due=a.get("due") or None,
            )
            for a in data.get("action_items", [])
            if a.get("task")
        ]

        return SummaryResult(
            summary=data.get("summary", "").strip(),
            key_points=[p.strip() for p in data.get("key_points", []) if p.strip()],
            action_items=action_items,
        )

    def _quick_summarize(self, chunk: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize this meeting transcript excerpt in 3-4 sentences, preserving any decisions or action items mentioned:\n\n{chunk}",
                }
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()


def get_summarizer(use_openai: bool = False):
    """Factory: returns an OpenAISummarizer if requested (and configured),
    otherwise falls back to the local HF-based summarizer."""
    if use_openai and os.getenv("OPENAI_API_KEY"):
        return OpenAISummarizer()
    return LocalSummarizer()
