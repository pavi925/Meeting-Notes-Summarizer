# 🎙️ Meeting / Lecture Notes Summarizer

Turn any audio recording (meeting, lecture, call) into **structured notes with
action items** — automatically.

```
Audio file  ─▶  Whisper (transcription)  ─▶  Summarization pipeline  ─▶  Structured notes
 (mp3/wav)      (faster-whisper)             (BART map-reduce +           (Markdown / JSON:
                                               FLAN-T5 action extractor)    summary, key points,
                                                                            action items, owners)
```

This is a **pipeline product**, not a training project: every model used
(Whisper, BART, FLAN-T5) is pretrained and pulled straight from Hugging Face /
OpenAI's Whisper. All the engineering here is in chaining them together
robustly, chunking long transcripts, extracting structured action items, and
wrapping it in a usable UI + CLI.

---

## ✨ Features

- 🎧 **Transcription** — `faster-whisper` (CTranslate2-optimized Whisper),
  runs on CPU or GPU, with timestamps.
- 📝 **Map-reduce summarization** — chunks long transcripts, summarizes each
  chunk with `facebook/bart-large-cnn`, then summarizes the summaries so
  arbitrarily long meetings still produce a tight overview.
- ✅ **Action item extraction** — `google/flan-t5-base` prompted to pull out
  concrete tasks, phrased as `[ ] Task — Owner (if mentioned) — Due (if mentioned)`.
- 🔑 **Key points / decisions** extracted separately from the general summary.
- 🌐 **Optional OpenAI mode** — drop in `OPENAI_API_KEY` to use GPT for
  summarization + action items instead of the local HF models (higher
  quality, needs internet + API key).
- 🖥️ **Streamlit web UI** — upload audio, click run, get notes + download
  buttons (Markdown & JSON).
- ⌨️ **CLI** — `python cli.py meeting.mp3 -o notes.md` for scripting /
  automation.

---

## 📁 Project structure

```
meeting-notes-summarizer/
├── app.py                  # Streamlit UI (main entry point for the web app)
├── cli.py                  # Command-line entry point
├── requirements.txt
├── .env.example             # copy to .env and fill in if using OpenAI mode
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── transcriber.py       # Whisper transcription wrapper
│   ├── summarizer.py        # Summarization + action item extraction
│   ├── pipeline.py          # Orchestrates transcribe -> summarize -> notes
│   └── utils.py             # Chunking, markdown formatting helpers
├── examples/
│   └── sample_output.md     # Example of generated notes
└── tests/
    └── test_pipeline.py     # Basic smoke tests (no audio required)
```

---

## 🚀 Quickstart

### 1. Clone & set up environment

```bash
git clone <your-repo-url>
cd meeting-notes-summarizer
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** `faster-whisper` needs `ffmpeg` installed on your system.
> - macOS: `brew install ffmpeg`
> - Ubuntu/Debian: `sudo apt install ffmpeg`
> - Windows: [download here](https://ffmpeg.org/download.html) and add to PATH.

### 2. (Optional) Enable OpenAI mode

```bash
cp .env.example .env
# then edit .env and add your OPENAI_API_KEY
```

If you skip this, the app runs 100% locally with Whisper + BART + FLAN-T5.

### 3. Run the web app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501), upload an
audio file, and click **Generate Notes**.

### 4. Or run from the command line

```bash
python cli.py path/to/meeting.mp3 -o notes.md --whisper-model base
```

Options:
```
python cli.py --help
```

---

## 🧠 How it works

1. **`src/transcriber.py`** loads a `faster-whisper` model (`tiny` / `base` /
   `small` / `medium` / `large-v3`, configurable) and transcribes the audio
   into text + segments with timestamps.
2. **`src/utils.py`** chunks the transcript into ~1000-word windows so it fits
   inside the summarizer's token limit, without cutting sentences in half.
3. **`src/summarizer.py`**:
   - Summarizes each chunk with `facebook/bart-large-cnn` (map step).
   - Summarizes the concatenation of chunk summaries again (reduce step) to
     get one coherent overview.
   - Runs `google/flan-t5-base` with a structured prompt to pull out action
     items, decisions, and key points as parsed lists.
   - If `OPENAI_API_KEY` is set and `--use-openai` / the UI toggle is
     enabled, this step is swapped for a single GPT call that returns
     structured JSON directly (usually higher quality).
4. **`src/pipeline.py`** ties it all together into a `MeetingNotes` object
   and renders it to Markdown / JSON.

---

## 📤 Pushing this to GitHub today

```bash
cd meeting-notes-summarizer
git init
git add .
git commit -m "Initial commit: meeting/lecture notes summarizer pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

(`.gitignore` already excludes `venv/`, `__pycache__/`, `.env`, and model
cache folders so you won't accidentally commit secrets or multi-GB model
weights.)

---

## ⚙️ Configuration reference

| Variable / flag        | Where            | Default              | Description |
|-------------------------|------------------|-----------------------|-------------|
| `OPENAI_API_KEY`        | `.env`           | unset                 | If set, enables `--use-openai` mode |
| `--whisper-model`       | CLI / UI sidebar | `base`                | tiny/base/small/medium/large-v3 |
| `--device`              | CLI / UI sidebar | `auto`                | cpu / cuda / auto |
| `--chunk-words`         | CLI              | `1000`                | Words per summarization chunk |

---

## 🛠️ Tech stack

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — transcription
- [transformers](https://github.com/huggingface/transformers) (`facebook/bart-large-cnn`, `google/flan-t5-base`) — summarization & extraction
- [Streamlit](https://streamlit.io/) — UI
- OpenAI API — optional higher-quality mode

## 📌 Roadmap ideas (not required for MVP)
- Speaker diarization (e.g. `pyannote.audio`)
- Slack/Notion export integrations
- Multi-language support (Whisper already supports it — just expose in UI)
