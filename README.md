markdown
<div align="center">

# 🤖 Smart RAG Chatbot

**A local, free, privacy-first RAG chatbot.**
Upload your documents, ask questions, get cited answers — no API key, no cloud, no data leaving your machine.

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-000000?logo=ollama&logoColor=white)](https://ollama.com/)
[![FAISS](https://img.shields.io/badge/FAISS-vector%20search-0467DF)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)

</div>

---

## ✨ Why this exists

Generic chatbots only know what's in their training data. Cloud RAG tools work
well, but your documents leave your machine the moment you upload them.

**Smart RAG Chatbot** splits the difference: it indexes your PDFs, DOCX, and
TXT files into a local vector store, retrieves the most relevant passages for
every question, and answers using a model running entirely on your own
hardware via [Ollama](https://ollama.com) — with every answer citing exactly
which document it came from.

No API key. No subscription. No data leaving your computer.

---

## 📚 Table of contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Project layout](#-project-layout)
- [Setup](#-setup)
- [Configuration](#-configuration-reference)
- [Usage](#-usage)
- [Roadmap](#-known-limitations--roadmap)
- [License](#-license)

---

## 🚀 Features

| | |
|---|---|
| 📄 **Multi-format ingestion** | Upload and index PDF, DOCX, and TXT files in one batch |
| 🔍 **Cited retrieval** | Every answer shows exactly which document(s) it drew from |
| 🔁 **Hot-swappable chat models** | Switch between mistral / llama3 / phi3 / and more mid-session — your indexed documents stay intact (embeddings and chat model are decoupled by design) |
| 🧵 **Conversation memory** | Configurable sliding-window memory keeps follow-up questions coherent |
| 🌗 **Polished UI** | Custom design system with light/dark mode, status badges, and toasts |
| 🔒 **Fully local & free** | Powered by Ollama + FAISS — nothing ever leaves your machine |

---

## 🏗️ Architecture

```
 Upload (PDF · DOCX · TXT)
            │
            ▼
 Chunk   RecursiveCharacterTextSplitter
            │
            ▼
 Embed   Ollama → EMBEDDING_MODEL  (e.g. nomic-embed-text)
            │
            ▼
 Index   FAISS vector store
            │
            ▼
 Question ──► Retrieve top-k chunks (MMR or similarity search)
            │
            ▼
 Generate   Ollama → DEFAULT_MODEL  (e.g. mistral)
            │
            ▼
      Answer + cited sources
```

> **Design note:** embeddings and the chat model are intentionally
> **decoupled**. Embeddings always use a fixed `EMBEDDING_MODEL`, while the
> chat model is whatever you pick in the sidebar. That means changing your
> chat model mid-conversation never invalidates the vector store — only the
> LLM and conversation chain are rebuilt.

---

## 🗂️ Project layout

```
smart-rag-chatbot/
├── Main.py              # Entry point — bootstrap, gates, module wiring
├── Session.py            # Single source of truth for st.session_state
├── Sidebar.py            # Branding, model picker, uploads, stats, controls
├── Chat.py               # Conversation rendering + prompt handling
├── Rag.py                # The RAG pipeline — chunk, embed, index, retrieve
├── Config.py              # Env-driven, validated app configuration
├── design/
│   ├── components.py     # Design system (cards, alerts, badges, tables…)
│   └── theme.py           # Color tokens / theming
├── static/
│   └── style.css          # Chrome styling
└── requirements.txt
```

| File | Responsibility |
|---|---|
| `Main.py` | Bootstraps the app, enforces Ollama availability gates, wires everything together |
| `Session.py` | All `st.session_state` reads/writes go through here |
| `Sidebar.py` | Model selection, document upload, stats, session controls |
| `Chat.py` | Renders chat history, handles a single conversational turn |
| `Rag.py` | Chunking, embedding, FAISS indexing, retrieval chain, inference |
| `Config.py` | Single source of truth for configuration — env-driven, validated at startup |

---

## ⚙️ Setup

### 1 · Install Ollama and pull the required models

```bash
ollama serve

# in another terminal
ollama pull mistral            # chat model — or llama3 / phi3 / gemma:2b / llama2 / codellama
ollama pull nomic-embed-text   # dedicated embedding model
```

### 2 · Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3 · Configure (optional — sensible defaults out of the box)

```bash
cp .env.example .env
```

### 4 · Run

```bash
streamlit run Main.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

---

## 🔧 Configuration reference

All settings are environment-driven via `.env` and validated at startup in `Config.py`.

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_MODEL` | `mistral` | Default chat model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Dedicated embedding model, independent of the chat model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `TEMPERATURE` | `0.2` | LLM sampling temperature (`0`–`2`) |
| `TOP_P` | `0.95` | Nucleus sampling parameter |
| `REPEAT_PENALTY` | `1.1` | Penalizes repeated tokens |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `RETRIEVAL_K` | `4` | Chunks retrieved per question |
| `RETRIEVAL_FETCH_K` | `8` | Candidate pool size before MMR re-ranking |
| `SEARCH_TYPE` | `mmr` | `mmr` (diverse results) or `similarity` |
| `MMR_LAMBDA` | `0.5` | Relevance/diversity trade-off for MMR (`0`–`1`) |
| `CONVERSATION_MEMORY_LENGTH` | `10` | Number of past turns retained in memory |
| `MAX_UPLOAD_SIZE_MB` | `200` | Max upload size per file |
| `DEBUG` | `False` | Verbose logging + chain trace |

---

## 💬 Usage

1. Start Ollama and confirm the sidebar shows **● Ollama running**.
2. Pick an installed model from the **Model** dropdown.
3. Upload one or more PDF / DOCX / TXT files under **Documents**, then click **Index documents**.
4. Ask a question in the chat box — answers will cite the source document(s) used.
5. No documents indexed yet? Just ask anyway — it falls back to the bare chat model.

---

## 🛣️ Known limitations / roadmap

- [ ] Vector store is in-memory per session — closing the tab loses the index (persistence to `VECTOR_STORE_PATH` is planned)
- [ ] No per-document deletion — only "Clear chat" or full "Reset all"
- [ ] Responses are not streamed token-by-token
- [ ] No automated test suite yet
- [ ] No Docker/compose setup yet

Contributions and suggestions welcome — open an issue or a PR.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

<div align="center">

Built with [Streamlit](https://streamlit.io) · [LangChain](https://www.langchain.com) · [Ollama](https://ollama.com) · [FAISS](https://github.com/facebookresearch/faiss)

</div>
```

