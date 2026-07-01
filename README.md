<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&pause=1000&color=1E88E5&center=true&vCenter=true&width=600&lines=🤖+Smart+RAG+Chatbot;Upload.+Ask.+Get+cited+answers." alt="Typing SVG" />

<br/>

**Upload your documents. Ask anything. Get cited answers — powered by Groq.**

<br/>

[![🚀 Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Try%20it%20now-1E88E5?style=for-the-badge)](https://smart-rag-chatbot-izxizdewwebpsrmhhufyxe.streamlit.app/)

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-Free%20Tier-F55036?style=flat-square)](https://groq.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0467DF?style=flat-square)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

</div>

---

## 📖 What is this?

Most chatbots only know what's in their training data. Most RAG tutorials either need a paid API or a local model server that takes 30 minutes to set up.

**Smart RAG Chatbot** is different. It:

- 📄 Indexes your **PDF, DOCX, and TXT** files into a local vector store
- 🔍 Retrieves the most relevant passages for every question
- ⚡ Answers via **Groq's hosted inference** — the fastest free LLM API available
- 📎 **Cites exactly which document** every answer came from

One `pip install`. One API key. No local model server. Just run it.

> Prefer fully offline? Set `LLM_PROVIDER=ollama` to use a local [Ollama](https://ollama.com) model instead — see [Configuration](#-configuration-reference).

---

## ✨ Features

<table>
<tr>
<td width="50%">

**🗂️ Smart Document Handling**
- Upload PDFs, DOCX, and TXT files in any batch
- Incremental indexing — new files added without re-indexing old ones
- Chunking with configurable overlap for context continuity

</td>
<td width="50%">

**🔍 Intelligent Retrieval**
- MMR search for diverse, non-redundant results
- Cited sources shown under every answer
- Conversation memory across turns

</td>
</tr>
<tr>
<td width="50%">

**⚡ Fast Inference**
- Groq: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `gemma2-9b-it`
- Hot-swap models mid-session — vector store stays intact
- Falls back to general chat when no documents are indexed

</td>
<td width="50%">

**🎨 Polished UI**
- Custom design system with light/dark mode
- Status badges, toasts, source cards
- Mobile-friendly responsive layout

</td>
</tr>
<tr>
<td width="50%">

**🔒 Privacy-First Embeddings**
- Document embeddings always run **on-device** via `sentence-transformers`
- Only the final chat prompt goes to Groq
- No document content is stored in any cloud

</td>
<td width="50%">

**🖥️ Offline Mode**
- Switch to `LLM_PROVIDER=ollama` for a 100% local setup
- Works with any Ollama model: `mistral`, `llama3`, `phi3`, `gemma:2b`

</td>
</tr>
</table>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Smart RAG Chatbot                        │
│                                                             │
│  📂 Upload (PDF · DOCX · TXT)                              │
│         │                                                   │
│         ▼                                                   │
│  ✂️  Chunk   RecursiveCharacterTextSplitter                 │
│         │                                                   │
│         ▼                                                   │
│  🧮  Embed   sentence-transformers/all-MiniLM-L6-v2        │
│         │         (always local — never sent to any API)   │
│         ▼                                                   │
│  📦  Index   FAISS vector store (in-memory)                │
│                                                             │
│  ❓ User question                                           │
│         │                                                   │
│         ├──► Retrieve  top-k chunks  (MMR search)          │
│         │                                                   │
│         ▼                                                   │
│  🤖  Generate  Groq API  ──  llama / gemma / ...           │
│         │         (or local Ollama if LLM_PROVIDER=ollama) │
│         ▼                                                   │
│  💬  Answer  +  📎 cited source documents                  │
└─────────────────────────────────────────────────────────────┘
```

> **Key design decision:** embeddings and the chat model are **intentionally decoupled**. Switching models or providers mid-session never invalidates the vector store — only the LLM and conversation chain are rebuilt.

---

## 🗂️ Project Structure

```
smart-rag-chatbot/
│
├── 🚀  app.py              # Streamlit Cloud entry point
├── 🧩  Main.py             # Bootstrap, provider gates, module wiring
│
├── 💬  Chat.py             # Conversation rendering + prompt handling
├── 📋  Session.py          # Single source of truth for st.session_state
├── 🎛️  Sidebar.py          # Model picker, uploads, stats, controls
│
├── 🧠  Rag.py              # RAG pipeline: chunk → embed → index → retrieve
├── ⚙️  Config.py           # Env-driven, validated app configuration
│
├── 🎨  design/
│   ├── components.py       # Design system: cards, alerts, badges, tables
│   └── theme.py            # Color tokens and dark/light mode theming
│
├── 🖼️  static/
│   └── style.css           # Chrome and layout styles
│
├── 📄  requirements.txt
├── 🔑  .env.example
└── 🔐  .streamlit/
    ├── config.toml         # Theme + server settings
    └── secrets.toml.example
```

---

## ⚙️ Setup

### Step 1 — Get a free Groq API key

Sign up at **[console.groq.com/keys](https://console.groq.com/keys)** — under a minute, no credit card required.

### Step 2 — Clone & install

```bash
git clone https://github.com/Junaid-Ahmed-Rupok/smart-rag-chatbot.git
cd smart-rag-chatbot

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> ⏱️ First install downloads PyTorch + sentence-transformers (~500MB). Subsequent installs are instant from cache.

### Step 3 — Configure

```bash
cp .env.example .env
```

Open `.env` and set your key:

```env
GROQ_API_KEY=gsk_your_key_here
```

### Step 4 — Run

```bash
streamlit run app.py
```

Open **[http://localhost:8501](http://localhost:8501)** — you're live. 🎉

The first question you ask will download the embedding model (~80MB, one-time only).

---

### 🖥️ Offline / Local Mode (optional)

To run **100% locally** with no API key:

```bash
# 1. Install Ollama from https://ollama.com
ollama serve
ollama pull mistral   # or: llama3 / phi3 / gemma:2b / codellama

# 2. Set in .env:
LLM_PROVIDER=ollama
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push your repo to GitHub
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → New app
3. Select your repo · Branch: `main` · Main file: `app.py`
4. Click **Advanced settings → Secrets** and add:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

5. Click **Deploy** — you'll get a public URL in ~60 seconds.

---

## 🔧 Configuration Reference

All settings live in `.env` (local) or Streamlit Secrets (cloud). Validated at startup in `Config.py`.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` (hosted) or `ollama` (local) |
| `GROQ_API_KEY` | — | **Required** for Groq. [Get one free](https://console.groq.com/keys) |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Active Groq chat model |
| `LOCAL_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Always local, regardless of provider |
| `DEFAULT_MODEL` | `mistral` | Default model when `LLM_PROVIDER=ollama` |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `TEMPERATURE` | `0.2` | Sampling temperature (`0`–`2`) |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_K` | `4` | Chunks retrieved per question |
| `RETRIEVAL_FETCH_K` | `8` | Candidate pool before MMR re-ranking |
| `SEARCH_TYPE` | `mmr` | `mmr` (diverse) or `similarity` |
| `MMR_LAMBDA` | `0.5` | Relevance vs diversity (`0`–`1`) |
| `CONVERSATION_MEMORY_LENGTH` | `10` | Past turns kept in memory |
| `MAX_UPLOAD_SIZE_MB` | `200` | Max file size per upload |
| `DEBUG` | `False` | Verbose logging + LangChain chain trace |

---

## 💬 Usage

```
1. Open the app — sidebar shows ● Groq connected
2. Pick a model from the Model dropdown
3. Upload PDF / DOCX / TXT files → click Index documents
4. Ask a question — answers cite the source document(s)
5. No documents? Just ask — falls back to general chat
```

**Sidebar controls:**
- **Clear chat** — wipes conversation history, keeps indexed documents
- **Reset all** — wipes everything including the vector store

---

## 🛣️ Roadmap

- [ ] Persistent vector store across sessions (FAISS save/load)
- [ ] Streaming token-by-token responses
- [ ] Per-document deletion from the index
- [ ] Docker / `docker-compose` setup
- [ ] Automated test suite

Contributions and suggestions welcome — open an issue or a PR.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

Built with ❤️ using

[Streamlit](https://streamlit.io) &nbsp;·&nbsp; [LangChain](https://www.langchain.com) &nbsp;·&nbsp; [Groq](https://groq.com) &nbsp;·&nbsp; [FAISS](https://github.com/facebookresearch/faiss) &nbsp;·&nbsp; [sentence-transformers](https://www.sbert.net)

<br/>

⭐ **If this helped you, consider starring the repo!** ⭐

</div>
