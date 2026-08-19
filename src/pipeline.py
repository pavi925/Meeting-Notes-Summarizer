"""
pipeline.py
-----------
Top-level orchestration: audio file -> Transcript -> SummaryResult ->
MeetingNotes (with Markdown/JSON export).
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable, Dict, List, Optional

from .transcriber import Transcriber, Transcript
from .summarizer import ActionItem, SummaryResult, get_summarizer
from .utils import notes_to_markdown


@dataclass
class MeetingNotes:
    title: str
    summary: str
    key_points: List[str]
    action_items: List[ActionItem]
    transcript: str
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_markdown(self) -> str:
        return notes_to_markdown(self)

    def to_json(self) -> str:
        payload = {
            "title": self.title,
            "summary": self.summary,
            "key_points": self.key_points,
            "action_items": [asdict(a) for a in self.action_items],
            "metadata": self.metadata,
            "transcript": self.transcript,
        }
        return json.dumps(payload, indent=2)


class MeetingNotesPipeline:
    """
    Ties transcription + summarization together.

    Usage:
        pipeline = MeetingNotesPipeline(whisper_model="base", use_openai=False)
        notes = pipeline.run("meeting.mp3", title="Weekly Sync")
        print(notes.to_markdown())
    """

    def __init__(
        self,
        whisper_model: str = "base",
        device: str = "auto",
        use_openai: bool = False,
        chunk_words: int = 1000,
    ):
        self.chunk_words = chunk_words
        self.use_openai = use_openai
        self._transcriber = Transcriber(model_size=whisper_model, device=device)
        self._summarizer = get_summarizer(use_openai=use_openai)

    def run(
        self,
        audio_path: str,
        title: Optional[str] = None,
        language: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> MeetingNotes:
        def transcribe_progress(pct: float):
            if progress_callback:
                progress_callback("transcribing", pct)

        transcript: Transcript = self._transcriber.transcribe(
            audio_path, language=language, progress_callback=transcribe_progress
        )

        if progress_callback:
            progress_callback("summarizing", 0.0)

        result: SummaryResult = self._summarizer.run(
            transcript.text, chunk_words=self.chunk_words
        )

        if progress_callback:
            progress_callback("done", 1.0)

        return MeetingNotes(
            title=title or "Meeting Notes",
            summary=result.summary,
            key_points=result.key_points,
            action_items=result.action_items,
            transcript=transcript.text,
            metadata={
                "Generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Duration": f"{(transcript.duration or 0) / 60:.1f} min",
                "Language": transcript.language or "unknown",
                "Backend": "OpenAI" if self.use_openai else "Local (BART + FLAN-T5)",
            },
        )
