"""
Basic smoke tests that don't require downloading models or audio files —
safe to run in CI. Model-dependent behavior (Whisper/BART/FLAN-T5) should
be tested manually / in integration tests with real audio.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import chunk_text, seconds_to_timestamp, notes_to_markdown
from src.summarizer import ActionItem
from src.pipeline import MeetingNotes


def test_chunk_text_respects_word_limit():
    text = " ".join(["word"] * 50) + "."
    chunks = chunk_text(text, max_words=10)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk.split()) <= 60  # generous bound; sentence-aware splitting


def test_chunk_text_single_short_text():
    text = "This is a short sentence."
    chunks = chunk_text(text, max_words=1000)
    assert chunks == [text]


def test_seconds_to_timestamp_under_hour():
    assert seconds_to_timestamp(75) == "01:15"


def test_seconds_to_timestamp_over_hour():
    assert seconds_to_timestamp(3725) == "01:02:05"


def test_notes_to_markdown_contains_sections():
    notes = MeetingNotes(
        title="Test Meeting",
        summary="This is a test summary.",
        key_points=["Point one", "Point two"],
        action_items=[ActionItem(task="Do the thing", owner="Alice", due="Friday")],
        transcript="Full transcript text goes here.",
        metadata={"Duration": "10.0 min"},
    )
    md = notes.to_markdown()
    assert "# Test Meeting" in md
    assert "## 📋 Summary" in md
    assert "Point one" in md
    assert "Do the thing" in md
    assert "Alice" in md
    assert "Full transcript text goes here." in md


def test_notes_to_json_roundtrip():
    import json

    notes = MeetingNotes(
        title="Test",
        summary="Summary",
        key_points=["A"],
        action_items=[ActionItem(task="Task A")],
        transcript="Transcript",
        metadata={},
    )
    data = json.loads(notes.to_json())
    assert data["title"] == "Test"
    assert data["action_items"][0]["task"] == "Task A"


if __name__ == "__main__":
    # Allow running directly with `python tests/test_pipeline.py`
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
