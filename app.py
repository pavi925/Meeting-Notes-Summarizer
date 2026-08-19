"""
app.py
------
Streamlit UI for the Meeting / Lecture Notes Summarizer.

Run with:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from src.pipeline import MeetingNotesPipeline

load_dotenv()

st.set_page_config(page_title="Meeting Notes Summarizer", page_icon="🎙️", layout="wide")

st.title("🎙️ Meeting / Lecture Notes Summarizer")
st.caption(
    "Upload a recording → get a transcript, summary, key points, and "
    "action items — powered by Whisper + BART/FLAN-T5 (or OpenAI)."
)

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("⚙️ Settings")

    whisper_model = st.selectbox(
        "Whisper model size",
        ["tiny", "base", "small", "medium", "large-v3"],
        index=1,
        help="Bigger = more accurate but slower. 'base' is a good default.",
    )

    device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)

    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    use_openai = st.toggle(
        "Use OpenAI for summarization",
        value=False,
        disabled=not has_openai_key,
        help=(
            "Requires OPENAI_API_KEY in your .env file. "
            "When off, everything runs locally with BART + FLAN-T5."
            if has_openai_key
            else "Set OPENAI_API_KEY in your .env file to enable this."
        ),
    )

    language = st.text_input(
        "Language code (optional)",
        value="",
        placeholder="e.g. en, es, fr — leave blank to auto-detect",
    )

    st.divider()
    st.markdown(
        "**Backend:** " + ("🌐 OpenAI" if use_openai else "💻 Local (BART + FLAN-T5)")
    )

# ----------------------------------------------------------------- main ---

title = st.text_input("Meeting / lecture title", value="Untitled Meeting")

uploaded_file = st.file_uploader(
    "Upload audio file", type=["mp3", "wav", "m4a", "mp4", "flac", "ogg"]
)

col_run, col_status = st.columns([1, 3])
run_clicked = col_run.button("🚀 Generate Notes", type="primary", disabled=uploaded_file is None)

if run_clicked and uploaded_file is not None:
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def progress_callback(stage: str, pct: float):
        if stage == "transcribing":
            status_text.info(f"🎧 Transcribing audio... {pct * 100:.0f}%")
            progress_bar.progress(pct * 0.6)
        elif stage == "summarizing":
            status_text.info("🧠 Summarizing transcript and extracting action items...")
            progress_bar.progress(0.7)
        elif stage == "done":
            status_text.success("✅ Done!")
            progress_bar.progress(1.0)

    try:
        with st.spinner("Loading models (first run downloads them, may take a while)..."):
            pipeline = MeetingNotesPipeline(
                whisper_model=whisper_model,
                device=device,
                use_openai=use_openai,
            )

        notes = pipeline.run(
            audio_path,
            title=title,
            language=language or None,
            progress_callback=progress_callback,
        )

        st.session_state["notes"] = notes

    except Exception as e:
        st.error(f"Something went wrong: {e}")
    finally:
        os.unlink(audio_path)

# ------------------------------------------------------------ results ----

if "notes" in st.session_state:
    notes = st.session_state["notes"]

    st.divider()
    meta_cols = st.columns(len(notes.metadata))
    for col, (key, value) in zip(meta_cols, notes.metadata.items()):
        col.metric(key, value)

    tab_summary, tab_actions, tab_transcript = st.tabs(
        ["📋 Summary & Key Points", "✅ Action Items", "📝 Full Transcript"]
    )

    with tab_summary:
        st.subheader("Summary")
        st.write(notes.summary)

        if notes.key_points:
            st.subheader("Key Points")
            for point in notes.key_points:
                st.markdown(f"- {point}")

    with tab_actions:
        if notes.action_items:
            for item in notes.action_items:
                owner = f" · **{item.owner}**" if item.owner else ""
                due = f" · due {item.due}" if item.due else ""
                st.checkbox(f"{item.task}{owner}{due}", key=item.task)
        else:
            st.info("No explicit action items were detected in this recording.")

    with tab_transcript:
        st.text_area("Transcript", notes.transcript, height=400)

    st.divider()
    dl_col1, dl_col2 = st.columns(2)
    dl_col1.download_button(
        "⬇️ Download notes.md",
        data=notes.to_markdown(),
        file_name="notes.md",
        mime="text/markdown",
    )
    dl_col2.download_button(
        "⬇️ Download notes.json",
        data=notes.to_json(),
        file_name="notes.json",
        mime="application/json",
    )
