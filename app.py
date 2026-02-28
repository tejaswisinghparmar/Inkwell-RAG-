"""
app.py — Inkwell: AI-Powered PDF Q&A

Upload any PDF and chat with your document instantly.
Everything runs in-memory — no external database needed.
Uses HuggingFace's free Inference API.

Deploy for free on Streamlit Community Cloud.

Usage (local):
    streamlit run app.py
"""

import hashlib
import io
import tempfile
from pathlib import Path

import streamlit as st
from gtts import gTTS
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Constants ───────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
CHUNK_SIZE_SMALL = 500
CHUNK_OVERLAP_SMALL = 100
TOP_K = 4

LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
STT_MODEL = "openai/whisper-large-v3-turbo"
HF_TOKEN = st.secrets["HF_TOKEN"]

# ── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Inkwell",
    page_icon="🖋️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Dark Aesthetic Theme ───────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:ital,wght@1,400;1,600&display=swap');

    .stApp {
        background: linear-gradient(160deg, #0d0d0d 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ──────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: rgba(13, 13, 13, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* ── Header ──────────────────────────────────────────────────── */
    .main-header {
        text-align: center;
        padding: 2rem 0 0.3rem 0;
    }
    .main-header h1 {
        font-family: 'Playfair Display', serif;
        font-style: italic;
        font-weight: 600;
        font-size: 3rem;
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: 0.02em;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.45);
        font-size: 0.95rem;
        font-weight: 300;
        margin-top: 0;
        letter-spacing: 0.03em;
    }

    /* ── Upload zone ─────────────────────────────────────────────── */
    .upload-zone {
        background: rgba(255, 255, 255, 0.025);
        border: 1px dashed rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem auto;
        max-width: 520px;
        text-align: center;
    }
    .upload-zone p {
        color: rgba(255,255,255,0.4);
        font-size: 0.85rem;
        margin: 0.5rem 0 0 0;
    }

    /* ── Chat messages ───────────────────────────────────────────── */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        padding: 0.8rem 0 !important;
    }

    /* ── Chat input ──────────────────────────────────────────────── */
    .stChatInput > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }
    .stChatInput textarea { color: #e0e0e0 !important; }

    /* ── Buttons ─────────────────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        letter-spacing: 0.03em;
        transition: all 0.3s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(118, 75, 162, 0.4) !important;
    }
    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #c0c0c0 !important;
        transition: all 0.2s ease;
    }

    /* ── File uploader (main page) ───────────────────────────────── */
    section[data-testid="stFileUploader"] {
        background: transparent;
        border: none;
        padding: 0;
    }

    /* ── Expander ────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 8px !important;
        color: rgba(255, 255, 255, 0.6) !important;
        font-size: 0.85rem !important;
    }

    /* ── Divider ─────────────────────────────────────────────────── */
    hr { border-color: rgba(255, 255, 255, 0.06) !important; }

    /* ── Scrollbar ───────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

    /* ── Progress bar ────────────────────────────────────────────── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
    }

    /* ── Caption ─────────────────────────────────────────────────── */
    .stCaption { color: rgba(255,255,255,0.35) !important; }

    /* ── Compact mic widget ───────────────────────────────────────── */
    [data-testid="stAudioInput"] {
        min-height: 0 !important;
    }
    [data-testid="stAudioInput"] > div {
        padding: 0 !important;
    }

    /* ── Hide Streamlit chrome ───────────────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent !important; }

    /* ── Sidebar text ────────────────────────────────────────────── */
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: rgba(255, 255, 255, 0.55);
        font-size: 0.85rem;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    /* ── Query log buttons in sidebar ────────────────────────────── */
    .query-log-btn {
        display: block;
        width: 100%;
        text-align: left;
        padding: 0.55rem 0.75rem;
        margin: 0.2rem 0;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 8px;
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.82rem;
        cursor: pointer;
        transition: all 0.2s ease;
        text-decoration: none;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .query-log-btn:hover {
        background: rgba(118, 75, 162, 0.15);
        border-color: rgba(118, 75, 162, 0.3);
        color: rgba(255, 255, 255, 0.85);
    }
    .query-log-btn.active {
        background: rgba(118, 75, 162, 0.2);
        border-color: rgba(118, 75, 162, 0.4);
        color: #e0c3fc;
    }

    /* ── Status pill ─────────────────────────────────────────────── */
    .status-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-left: 0.5rem;
    }
    .status-ready {
        background: rgba(46, 213, 115, 0.15);
        color: #2ed573;
        border: 1px solid rgba(46, 213, 115, 0.25);
    }
</style>
""", unsafe_allow_html=True)


# ── Cached Embedding Model ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def get_embedding_model():
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


# ── PDF Processing with Progress ───────────────────────────────────
def process_pdf(uploaded_file, embedding_model):
    """Load PDF -> chunk -> embed -> return in-memory vector store."""
    progress = st.progress(0, text="Loading PDF...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        docs = PyPDFLoader(file_path=tmp_path).load()
    except Exception as e:
        Path(tmp_path).unlink(missing_ok=True)
        progress.empty()
        raise ValueError(f"Invalid or corrupted PDF file: {uploaded_file.name}") from e

    Path(tmp_path).unlink(missing_ok=True)

    if not docs:
        progress.empty()
        raise ValueError(f"Could not extract any pages from: {uploaded_file.name}")

    progress.progress(20, text=f"Loaded {len(docs)} pages. Chunking...")

    # Measure total text to pick chunking strategy
    total_text = sum(len(d.page_content) for d in docs)

    if total_text < 2000:
        # Small document — use smaller chunks so we get meaningful splits
        chunk_size, chunk_overlap = CHUNK_SIZE_SMALL, CHUNK_OVERLAP_SMALL
    else:
        chunk_size, chunk_overlap = CHUNK_SIZE, CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        # If splitting produced nothing, use the raw pages as chunks
        chunks = docs

    progress.progress(40, text=f"Created {len(chunks)} chunks. Embedding...")

    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        location=":memory:",
        collection_name="uploaded_pdf",
    )
    progress.progress(100, text="Done!")
    return vector_store, len(docs), len(chunks)


# ── Text-to-Speech ──────────────────────────────────────────────────
def text_to_speech(text: str):
    """Convert text to an in-memory MP3 audio buffer using gTTS."""
    tts = gTTS(text=text, lang="en")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf


# ── Speech-to-Text ──────────────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio bytes using HuggingFace Whisper model."""
    # Save to a temp .wav file so HuggingFace gets the correct content type
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        client = InferenceClient(token=HF_TOKEN)
        result = client.automatic_speech_recognition(
            tmp_path, model=STT_MODEL
        )
        return result.text if hasattr(result, "text") else str(result)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── RAG Pipeline ────────────────────────────────────────────────────
def retrieve_and_generate(query: str, vector_db, model_id: str):
    """Retrieve -> Prompt -> Generate using HuggingFace Inference API."""
    # Use fewer results if the collection is small
    k = min(TOP_K, max(1, vector_db._client.count("uploaded_pdf").count))
    results = vector_db.similarity_search(query=query, k=k)

    context_parts, sources = [], []
    for r in results:
        page = r.metadata.get("page_label", r.metadata.get("page", "N/A"))
        context_parts.append(f"Page Content: {r.page_content}\nPage Number: {page}")
        sources.append(f"Page {page}")

    context = "\n\n".join(context_parts)

    system_msg = (
        "You are a helpful AI Assistant. Answer the user's question ONLY "
        "based on the context below. If the context doesn't contain the "
        "answer, say you don't know. Always cite the relevant Page Number."
    )
    user_msg = f"Context:\n{context}\n\nUser Query: {query}"

    client = InferenceClient(token=HF_TOKEN)
    response = client.chat_completion(
        model=model_id,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=1024,
        temperature=0.3,
    )
    answer = response.choices[0].message.content
    return answer, list(dict.fromkeys(sources))


# ── Session State Init ──────────────────────────────────────────────
for key, default in {
    "messages": [],
    "vector_db": None,
    "pdf_name": None,
    "page_count": 0,
    "chunk_count": 0,
    "scroll_to": None,
    "last_audio_hash": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Header ──────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>Inkwell</h1>
    <p>Upload a PDF and start a conversation with your document</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar — Query Log ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Query History")

    # Collect all user queries from messages
    user_queries = [
        (i, msg["content"])
        for i, msg in enumerate(st.session_state.messages)
        if msg["role"] == "user"
    ]

    if user_queries:
        for msg_idx, query_text in user_queries:
            truncated = query_text[:50] + ("..." if len(query_text) > 50 else "")
            if st.button(
                f"💬 {truncated}",
                key=f"qlog_{msg_idx}",
                use_container_width=True,
            ):
                st.session_state.scroll_to = msg_idx
                st.rerun()
    else:
        st.markdown(
            '<p style="color: rgba(255,255,255,0.3); font-size: 0.82rem;">'
            "No queries yet. Upload a PDF and ask a question.</p>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Document status
    if st.session_state.pdf_name:
        st.markdown(
            f'### Active Document\n\n'
            f'**{st.session_state.pdf_name}**  \n'
            f'{st.session_state.page_count} pages · '
            f'{st.session_state.chunk_count} chunks'
        )
    else:
        st.markdown(
            '<p style="color: rgba(255,255,255,0.3); font-size: 0.82rem;">'
            "No document loaded.</p>",
            unsafe_allow_html=True,
        )

    st.divider()

    if st.session_state.messages:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.scroll_to = None
            st.rerun()


# ── Main Area — Upload or Chat ──────────────────────────────────────
if not st.session_state.vector_db:
    # ── Upload Zone (main page) ─────────────────────────────────────
    st.markdown("")  # spacing

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "Upload a PDF to get started",
            type=["pdf"],
            help="Processed in-memory only — never stored.",
            label_visibility="visible",
        )

        if uploaded_file:
            if st.button("Process & Index", use_container_width=True, type="primary"):
                try:
                    embedding_model = get_embedding_model()
                    vector_db, pages, chunks = process_pdf(uploaded_file, embedding_model)
                    st.session_state.vector_db = vector_db
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.page_count = pages
                    st.session_state.chunk_count = chunks
                    st.session_state.messages = []
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Failed to process PDF: {e}")
        else:
            st.markdown(
                '<p style="color: rgba(255,255,255,0.35); font-size: 0.82rem; text-align: center;">'
                "100% free — no sign-up, no data stored.</p>",
                unsafe_allow_html=True,
            )

else:
    # ── Chat Interface ──────────────────────────────────────────────
    st.caption(f"Chatting with {st.session_state.pdf_name}")

    # Display chat history with anchor IDs for scroll-to
    for i, msg in enumerate(st.session_state.messages):
        # Inject scroll anchor for each message
        st.markdown(f'<div id="msg-{i}"></div>', unsafe_allow_html=True)

        avatar = "🧑‍💻" if msg["role"] == "user" else "🖋️"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    st.markdown(" · ".join(msg["sources"]))
            # TTS: listen button for assistant responses
            if msg["role"] == "assistant":
                if st.button("🔊 Listen", key=f"tts_{i}", help="Read aloud"):
                    try:
                        audio_buf = text_to_speech(msg["content"])
                        st.audio(audio_buf, format="audio/mp3")
                    except Exception:
                        st.caption("Audio generation failed.")

    # Scroll to a specific message if requested from sidebar
    if st.session_state.scroll_to is not None:
        target = st.session_state.scroll_to
        st.markdown(
            f"""
            <script>
                const el = document.getElementById('msg-{target}');
                if (el) {{ el.scrollIntoView({{ behavior: 'smooth', block: 'start' }}); }}
            </script>
            """,
            unsafe_allow_html=True,
        )
        st.session_state.scroll_to = None

    # ── Voice + Text Input Row ───────────────────────────────────────
    mic_col, _ = st.columns([1, 5])
    with mic_col:
        audio_data = st.audio_input("Voice", label_visibility="collapsed")
    if audio_data:
        audio_hash = hashlib.md5(audio_data.getvalue()).hexdigest()
        if st.session_state.last_audio_hash != audio_hash:
            st.session_state.last_audio_hash = audio_hash
            with st.spinner("Transcribing your voice..."):
                try:
                    transcribed = transcribe_audio(audio_data.getvalue())
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    transcribed = None
            if transcribed and transcribed.strip():
                voice_prompt = transcribed.strip()
                st.session_state.messages.append(
                    {"role": "user", "content": voice_prompt}
                )
                with st.spinner("Thinking..."):
                    try:
                        answer, sources = retrieve_and_generate(
                            voice_prompt,
                            st.session_state.vector_db,
                            LLM_MODEL,
                        )
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer, "sources": sources}
                        )
                    except Exception as e:
                        err = str(e)
                        if "401" in err or "Unauthorized" in err or "Invalid" in err:
                            error_msg = "Authentication error. Please try again later."
                        elif "loading" in err.lower() or "unavailable" in err.lower():
                            error_msg = "Model is waking up (~30s). Try again in a moment."
                        else:
                            error_msg = f"Something went wrong: {err}\n\nPlease try again."
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )
                st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask something about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🖋️"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = retrieve_and_generate(
                        prompt, st.session_state.vector_db, LLM_MODEL
                    )
                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            st.markdown(" · ".join(sources))
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as e:
                    err = str(e)
                    if "401" in err or "Unauthorized" in err or "Invalid" in err:
                        error_msg = "Authentication error. Please try again later."
                    elif "loading" in err.lower() or "unavailable" in err.lower():
                        error_msg = "Model is waking up (~30s). Try again in a moment."
                    else:
                        error_msg = f"Something went wrong: {err}\n\nPlease try again."
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

    # Option to upload a new PDF
    with st.expander("Upload a different PDF"):
        new_file = st.file_uploader(
            "Replace document",
            type=["pdf"],
            key="new_pdf_uploader",
            label_visibility="collapsed",
        )
        if new_file and new_file.name != st.session_state.pdf_name:
            if st.button("Process New PDF", type="primary", use_container_width=True):
                try:
                    embedding_model = get_embedding_model()
                    vector_db, pages, chunks = process_pdf(new_file, embedding_model)
                    st.session_state.vector_db = vector_db
                    st.session_state.pdf_name = new_file.name
                    st.session_state.page_count = pages
                    st.session_state.chunk_count = chunks
                    st.session_state.messages = []
                    st.session_state.scroll_to = None
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Failed to process PDF: {e}")
