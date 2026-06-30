markdown
<div align="center">

# 🤖 Smart RAG Chatbot

**A fast, friction-free RAG chatbot.**
Upload your documents, ask questions, get cited answers — just `pip install` and run, no local model server to set up.

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-fast%20inference-F55036)](https://groq.com/)
[![FAISS](https://img.shields.io/badge/FAISS-vector%20search-0467DF)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)

</div>

---

## ✨ Why this exists

Generic chatbots only know what's in their training data. Most RAG tutorials
either require a paid cloud subscription, or a local LLM server you have to
install, configure, and keep running.

**Smart RAG Chatbot** is built for end users, not just developers: it indexes
your PDFs, DOCX, and TXT files into a local vector store, retrieves the most
relevant passages for every question, and answers using [Groq](https://groq.com)'s
hosted inference — one of the fastest LLM APIs available, with a generous free
tier. Every answer cites exactly which document it came from.

One `pip install`, one API key, and it runs like any other professional chatbot.
(Prefer fully offline/local? It also supports [Ollama](https://ollama.com) — see
[Configuration](#-configuration-reference).)

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
| ⚡ **Fast, hosted inference** | Powered by Groq — no local model server, just an API key |
| 🔁 **Hot-swappable chat models** | Switch between Groq models mid-session — your indexed documents stay intact (embeddings and chat model are decoupled by design) |
| 🧵 **Conversation memory** | Configurable sliding-window memory keeps follow-up questions coherent |
| 🌗 **Polished UI** | Custom design system with light/dark mode, status badges, and toasts |
| 🔒 **Local embeddings** | Document embeddings always run on-device (sentence-transformers) — only the chat turn goes to Groq |
| 🖥️ **Offline mode available** | Set `LLM_PROVIDER=ollama` for a fully local, no-API-key setup |

---

## 🏗️ Architecture

```
 Upload (PDF · DOCX · TXT)
            │
            ▼
 Chunk   RecursiveCharacterTextSplitter
            │
            ▼
 Embed   Local sentence-transformers model (always on-device)
            │
            ▼
 Index   FAISS vector store
            │
            ▼
 Question ──► Retrieve top-k chunks (MMR or similarity search)
            │
            ▼
 Generate   Groq API → GROQ_MODEL  (e.g. llama-3.1-8b-instant)
            │              — or local Ollama if LLM_PROVIDER=ollama
            ▼
      Answer + cited sources
```

> **Design note:** embeddings and the chat model are intentionally
> **decoupled**. Embeddings always run locally via a fixed
> `LOCAL_EMBEDDING_MODEL`, while the chat model is whatever you pick in the
> sidebar. That means changing your chat model — or even switching between
> Groq and Ollama — never invalidates the vector store; only the LLM and
> conversation chain are rebuilt.

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
| `Main.py` | Bootstraps the app, enforces the active provider's readiness gates, wires everything together |
| `Session.py` | All `st.session_state` reads/writes go through here |
| `Sidebar.py` | Model selection, document upload, stats, session controls |
| `Chat.py` | Renders chat history, handles a single conversational turn |
| `Rag.py` | Chunking, embedding, FAISS indexing, retrieval chain, inference |
| `Config.py` | Single source of truth for configuration — env-driven, validated at startup |

---

## ⚙️ Setup

### 1 · Get a free Groq API key

Sign up at [console.groq.com/keys](https://console.groq.com/keys) and create a key — takes under a minute, no credit card required.

### 2 · Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3 · Configure

```bash
cp .env.example .env
```

Open `.env` and paste your key into `GROQ_API_KEY=`.

### 4 · Run

```bash
streamlit run Main.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

That's it — no separate model server to install or keep running. The first
question you ask will download the local embedding model (~80MB, one-time).

> **Want it fully offline instead?** Set `LLM_PROVIDER=ollama` in `.env`,
> then:
> ```bash
> ollama serve
> ollama pull mistral   # or llama3 / phi3 / gemma:2b / llama2 / codellama
> ```

---

## 🔧 Configuration reference

All settings are environment-driven via `.env` and validated at startup in `Config.py`.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` (hosted, no install) or `ollama` (fully local) |
| `GROQ_API_KEY` | — | **Required** if `LLM_PROVIDER=groq`. Get one at [console.groq.com/keys](https://console.groq.com/keys) |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Default Groq chat model |
| `LOCAL_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model — always local, regardless of `LLM_PROVIDER` |
| `DEFAULT_MODEL` | `mistral` | Default chat model when `LLM_PROVIDER=ollama` |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL (only used when `LLM_PROVIDER=ollama`) |
| `TEMPERATURE` | `0.2` | LLM sampling temperature (`0`–`2`) |
| `TOP_P` | `0.95` | Nucleus sampling parameter |
| `REPEAT_PENALTY` | `1.1` | Penalizes repeated tokens (Ollama only) |
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

1. Start the app — the sidebar shows **● Groq connected** once your key is verified.
2. Pick a model from the **Model** dropdown.
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

Built with [Streamlit](https://streamlit.io) · [LangChain](https://www.langchain.com) · [Groq](https://groq.com) · [FAISS](https://github.com/facebookresearch/faiss)

</div>
```

