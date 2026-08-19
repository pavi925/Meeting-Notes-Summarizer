"""
utils.py
--------
Small stateless helpers used by the pipeline: text chunking and
Markdown rendering.
"""

import re
from typing import List


def chunk_text(text: str, max_words: int = 1000) -> List[str]:
    """
    Split text into chunks of at most `max_words` words, breaking on
    sentence boundaries so summarization doesn't cut a sentence in half.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: List[str] = []
    current: List[str] = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_word_count + word_count > max_words and current:
            chunks.append(" ".join(current))
            current = [sentence]
            current_word_count = word_count
        else:
            current.append(sentence)
            current_word_count += word_count

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds -> HH:MM:SS (or MM:SS if under an hour)."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def notes_to_markdown(notes) -> str:
    """
    Render a MeetingNotes object (see src/pipeline.py) into a clean
    Markdown document.
    """
    lines = []
    lines.append(f"# {notes.title}\n")

    if notes.metadata:
        meta_bits = [f"**{k}:** {v}" for k, v in notes.metadata.items()]
        lines.append(" | ".join(meta_bits) + "\n")

    lines.append("## 📋 Summary\n")
    lines.append(notes.summary.strip() + "\n")

    if notes.key_points:
        lines.append("## 🔑 Key Points\n")
        for point in notes.key_points:
            lines.append(f"- {point}")
        lines.append("")

    if notes.action_items:
        lines.append("## ✅ Action Items\n")
        for item in notes.action_items:
            owner = f" — **{item.owner}**" if item.owner else ""
            due = f" (due: {item.due})" if item.due else ""
            lines.append(f"- [ ] {item.task}{owner}{due}")
        lines.append("")
    else:
        lines.append("## ✅ Action Items\n")
        lines.append("_No explicit action items were detected._\n")

    lines.append("## 📝 Full Transcript\n")
    lines.append("<details><summary>Click to expand</summary>\n")
    lines.append(notes.transcript.strip())
    lines.append("\n</details>\n")

    return "\n".join(lines)
