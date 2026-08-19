"""
transcriber.py
--------------
Thin wrapper around faster-whisper for turning an audio file into a
transcript (full text + timestamped segments).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable

from faster_whisper import WhisperModel


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    text: str
    segments: List[TranscriptSegment] = field(default_factory=list)
    language: Optional[str] = None
    duration: Optional[float] = None


class Transcriber:
    """
    Loads a faster-whisper model once and reuses it across calls.

    Parameters
    ----------
    model_size : str
        One of "tiny", "base", "small", "medium", "large-v3" (bigger = more
        accurate, slower). "base" is a good default for CPU.
    device : str
        "cpu", "cuda", or "auto" (auto-detects GPU if available).
    compute_type : str
        Precision used by CTranslate2. "int8" is fastest on CPU,
        "float16" is a good choice on GPU.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: Optional[str] = None,
    ):
        resolved_device = self._resolve_device(device)
        if compute_type is None:
            compute_type = "float16" if resolved_device == "cuda" else "int8"

        self.model = WhisperModel(
            model_size,
            device=resolved_device,
            compute_type=compute_type,
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Transcript:
        """
        Transcribe an audio file.

        progress_callback, if given, is called with a float in [0, 1]
        as segments are produced (based on elapsed audio time vs total
        duration) -- handy for driving a UI progress bar.
        """
        segments_iter, info = self.model.transcribe(
            audio_path,
            language=language,
            vad_filter=True,  # skip silence, improves quality/speed
            beam_size=5,
        )

        segments: List[TranscriptSegment] = []
        full_text_parts: List[str] = []
        total_duration = info.duration or 0.0

        for seg in segments_iter:
            segments.append(
                TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip())
            )
            full_text_parts.append(seg.text.strip())

            if progress_callback and total_duration > 0:
                progress_callback(min(seg.end / total_duration, 1.0))

        return Transcript(
            text=" ".join(full_text_parts).strip(),
            segments=segments,
            language=info.language,
            duration=total_duration,
        )
