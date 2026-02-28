# Inkwell — AI-Powered PDF Q&A

> **Upload any PDF and ask questions** — get accurate, page-cited answers powered by Retrieval-Augmented Generation (RAG) with a free HuggingFace LLM, FastEmbed, and in-memory Qdrant.

**[Live Demo](https://inkwell-rag.streamlit.app/) · [GitHub](https://github.com/tejaswisinghparmar/Inkwell-RAG-)**

---

## Key Features

| Feature | Details |
|---|---|
| **Upload Any PDF** | Upload on the main page — no pre-indexed data needed |
| **Zero Setup for Users** | No API keys, no sign-up — just upload and chat |
| **In-Memory Processing** | PDF is chunked, embedded, and stored in RAM — nothing persists after the session |
| **Smart Chunking** | Recursive character splitting (1 500 chars, 300-char overlap) preserves context across pages |
| **Fast Embeddings** | `BAAI/bge-small-en-v1.5` (384-dim) via FastEmbed — runs on CPU, zero API cost |
| **Page Citations** | Every answer references the exact page number for verification |
| **Query History** | Sidebar logs every question — click any query to scroll to that Q&A |
| **Dark Aesthetic UI** | Gradient background, Playfair Display italic heading, glassmorphism sidebar |
| **Privacy-First** | No data stored, no keys exposed — everything dies when you close the tab |
| **CLI Tools** | Bonus: CLI scripts for local batch indexing and terminal-based chat |

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │         Streamlit Web App (Inkwell)          │
                    └──────────────────────────────────────────────┘
                                        │
                     ┌──────────────────┼──────────────────┐
                     ▼                  ▼                  ▼
              ┌────────────┐   ┌──────────────┐   ┌──────────────┐
  User ──▶   │  Upload PDF │   │  User Query   │   │ Query History │
              └─────┬──────┘   └──────┬───────┘   │  (Sidebar)   │
                    │                 │            └──────────────┘
                    ▼                 ▼
           ┌──────────────┐   ┌──────────────────┐
           │  PyPDF Load   │   │   Embed Query     │
           │  + Chunking   │   │  (FastEmbed/CPU)   │
           └──────┬───────┘   └────────┬──────────┘
                  │                    │
                  ▼                    ▼
           ┌──────────────┐   ┌──────────────────┐
           │   FastEmbed   │   │  Cosine Search    │
           │  (bge-small)  │   │  Qdrant In-Memory │
           └──────┬───────┘   └────────┬──────────┘
                  │                    │
                  ▼                    ▼
           ┌──────────────┐   ┌────────────────────────────────────┐
           │ Qdrant Store  │   │  System Prompt + Context + Query   │
           │  (in-memory)  │   │      → Qwen 2.5 7B (HuggingFace)  │
           └──────────────┘   └──────────────┬─────────────────────┘
                                             │
                                             ▼
                                    ┌──────────────┐
                                    │   Answer +    │
                                    │  Page Cited   │
                                    └──────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Framework | LangChain |
| Frontend | Streamlit (dark themed, custom CSS) |
| Vector DB | Qdrant — in-memory (`:memory:`) for web app, Docker for CLI |
| Embeddings | FastEmbed — `BAAI/bge-small-en-v1.5` (384-dim, runs on CPU, zero cost) |
| LLM | `Qwen/Qwen2.5-7B-Instruct` via HuggingFace free Inference API (~1 000 req/day) |
| PDF Loader | PyPDF |
| Chunking | `RecursiveCharacterTextSplitter` — 1 500 chars, 300-char overlap |

---

## Project Structure

```
.
├── app.py                    # Streamlit web UI — main application
├── rag/
│   ├── index.py              # CLI: Ingest PDF → chunk → embed → Qdrant
│   ├── chat.py               # CLI: Single-query RAG chat
│   ├── chat_autorun.py       # CLI: Interactive multi-turn chat loop
│   └── docker-compose.yml    # One-command Qdrant setup (for CLI mode)
├── .streamlit/
│   ├── config.toml           # Streamlit theme config (dark mode)
│   └── secrets.toml          # HF token (local only, gitignored)
├── .env.example              # Template for CLI environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quick Start

### Option A: Web UI (Recommended)

```bash
git clone https://github.com/tejaswisinghparmar/Inkwell-RAG-.git
cd Inkwell-RAG-
python -m venv venv && source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:
```toml
HF_TOKEN = "your_huggingface_token_here"
```

Run:
```bash
streamlit run app.py
```

Open **http://localhost:8501** → upload a PDF → start chatting.

### Option B: CLI Mode (with Docker Qdrant)

```bash
# 1. Start Qdrant
cd rag && docker compose up -d && cd ..

# 2. Configure
cp .env.example rag/.env
# Edit rag/.env with your GOOGLE_API_KEY

# 3. Index a PDF
python rag/index.py

# 4. Chat
python rag/chat_autorun.py
```

---

## Free Deployment (Streamlit Community Cloud)

The web app needs **zero external services** — no database, no paid APIs from users.

| What | Where | Cost |
|---|---|---|
| Web App | [Streamlit Community Cloud](https://streamlit.io/cloud) | Free |
| LLM | HuggingFace Inference API (Qwen 2.5 7B) | Free (~1 000 req/day) |
| Vector DB | In-memory Qdrant (no setup) | Free |
| Embeddings | FastEmbed on CPU | Free |

### Deploy in 3 Steps

1. **Push to GitHub** — public repo
2. **[share.streamlit.io](https://share.streamlit.io/)** → connect repo → set main file to `app.py`
3. **Add secret** → Settings → Secrets → paste `HF_TOKEN = "hf_..."` → Save
4. **Done!** Share the URL on your resume.

---

## Security & Privacy

- **Server-side key** — HF token is stored in Streamlit secrets, never exposed to users.
- **No persistent storage** — PDFs processed in-memory and discarded when the session ends.
- **`.gitignore` protection** — `secrets.toml`, `.env` files, and PDFs excluded from version control.
- **Users see nothing** — no API key input, no tokens in the browser.

---

## How It Works

1. **Upload** — User uploads a PDF on the main page.
2. **Chunk** — The PDF is split into overlapping chunks (1 500 chars each, 300-char overlap) using `RecursiveCharacterTextSplitter` to preserve context across page boundaries.
3. **Embed** — Each chunk is embedded using `BAAI/bge-small-en-v1.5` (384-dim vectors) running locally on CPU via FastEmbed.
4. **Store** — Vectors are stored in an in-memory Qdrant instance (no external database).
5. **Retrieve** — When the user asks a question, the query is embedded and the top-4 most similar chunks are retrieved via cosine similarity.
6. **Generate** — Retrieved chunks + page numbers are injected into a system prompt, and `Qwen 2.5 7B Instruct` generates a grounded, cited answer via HuggingFace's free Inference API.
7. **Log** — Every query appears in the sidebar history — click to scroll back to any Q&A.

---

## Challenges & Learnings

- **Chunking strategy matters** — Recursive splitting with 300-char overlap significantly improved retrieval accuracy for questions spanning multiple pages.
- **FastEmbed vs cloud embeddings** — Switched from cloud-based embedding APIs to FastEmbed for zero-cost, offline-capable, faster indexing on CPU.
- **LLM selection** — Tried Google Gemini (rate-limited), then multiple HuggingFace models. Qwen 2.5 7B Instruct was the most reliable free option.
- **Prompt engineering for citations** — Explicitly instructing the LLM to cite page numbers reduced hallucinated answers and improved verifiability.
- **In-memory Qdrant** — Using `:memory:` mode eliminates the need for an external database while keeping the same LangChain API.
- **Sidebar as query log** — Moving PDF upload to the main page freed the sidebar for a clickable query history.

---

## Future Scope

- **Agentic RAG** — Tool-calling to let the LLM decide when to search, summarise, or ask for clarification
- **Hybrid Retrieval** — Dense (vector) + sparse (BM25) search for better recall
- **Reranking** — Cross-encoder reranker (e.g., `ms-marco-MiniLM`) for improved precision
- **Multi-document support** — Upload multiple PDFs and filter by source at retrieval
- **Streaming responses** — Token-by-token output for a more responsive feel
- **Evaluation with RAGAS** — Measure faithfulness, answer relevance, and context precision
- **Chat history export** — Download conversation as PDF/Markdown

---

## License

This project is open-source under the [MIT License](LICENSE).

---

> Built with LangChain, Qdrant, FastEmbed, HuggingFace, and Streamlit.
