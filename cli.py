"""
cli.py
------
Command-line interface for the Meeting / Lecture Notes Summarizer.

Usage:
    python cli.py meeting.mp3
    python cli.py lecture.wav -o notes.md --whisper-model small
    python cli.py call.mp3 --use-openai --json
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from src.pipeline import MeetingNotesPipeline

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Generate structured meeting/lecture notes from an audio file."
    )
    parser.add_argument("audio", help="Path to the audio file (mp3, wav, m4a, ...)")
    parser.add_argument(
        "-o", "--output", default=None, help="Output file path (default: notes.md next to audio)"
    )
    parser.add_argument("-t", "--title", default=None, help="Title for the notes")
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--language", default=None, help="Force transcription language (e.g. en, es)"
    )
    parser.add_argument(
        "--chunk-words", type=int, default=1000, help="Words per summarization chunk"
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI for summarization instead of local BART/FLAN-T5 "
        "(requires OPENAI_API_KEY in .env)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Also write a .json version of the notes"
    )

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    if args.use_openai and not os.getenv("OPENAI_API_KEY"):
        print(
            "Warning: --use-openai passed but OPENAI_API_KEY is not set. "
            "Falling back to local models.",
            file=sys.stderr,
        )

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.audio)), "notes.md"
    )
    title = args.title or os.path.splitext(os.path.basename(args.audio))[0]

    print(f"Loading models (whisper={args.whisper_model}, device={args.device})...")
    pipeline = MeetingNotesPipeline(
        whisper_model=args.whisper_model,
        device=args.device,
        use_openai=args.use_openai,
        chunk_words=args.chunk_words,
    )

    def progress_callback(stage: str, pct: float):
        if stage == "transcribing":
            print(f"\rTranscribing... {pct * 100:5.1f}%", end="", flush=True)
        elif stage == "summarizing":
            print("\nSummarizing and extracting action items...")
        elif stage == "done":
            print("Done.")

    notes = pipeline.run(
        args.audio,
        title=title,
        language=args.language,
        progress_callback=progress_callback,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(notes.to_markdown())
    print(f"\nMarkdown notes written to: {output_path}")

    if args.json:
        json_path = os.path.splitext(output_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(notes.to_json())
        print(f"JSON notes written to: {json_path}")


if __name__ == "__main__":
    main()
