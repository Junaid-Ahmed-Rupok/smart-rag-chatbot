
<div align="center">

# 🤖 Smart RAG Chatbot

**Upload your documents. Ask questions. Get cited answers.**
No local model server. No complex setup. Just one API key and you're running.

<br/>

[![Live Demo](https://img.shields.io/badge/Live%20Demo-%F0%9F%9A%80%20Try%20it%20now-1E88E5?style=for-the-badge)](https://smart-rag-chatbot-izxizdewwebpsrmhhufyxe.streamlit.app/)

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-Free%20Tier-F55036)](https://groq.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0467DF)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E)](LICENSE)

</div>

---

## ✨ Why this exists

Most chatbots only know what's in their training data. Most RAG tutorials either need a paid cloud subscription, or a local LLM server that takes 30 minutes to configure.

**Smart RAG Chatbot** is built for everyone — not just developers. It indexes your PDFs, DOCX, and TXT files into a local vector store, retrieves the most relevant passages for every question, and answers using [Groq](https://groq.com)'s hosted inference — one of the fastest free LLM APIs available. Every answer cites exactly which document it came from.

One `pip install`. One API key. No local model server.

> Prefer fully offline? Set `LLM_PROVIDER=ollama` to use a local [Ollama](https://ollama.com) model — see [Configuration](#-configuration-reference).

---

## 🚀 Features

| | |
|---|---|
| 📄 **Multi-format ingestion** | Upload and index PDF, DOCX, and TXT files in one batch |
| 🔍 **Cited retrieval** | Every answer shows exactly which document(s) it drew from |
| ⚡ **Fast, hosted inference** | Powered by Groq — no local model server, just an API key |
| 🔁 **Hot-swappable models** | Switch Groq models mid-session without discarding your indexed documents |
| 🧵 **Conversation memory** | Sliding-window memory keeps follow-up questions coherent |
| 🌗 **Polished UI** | Custom design system with light/dark mode, status badges, and toasts |
| 🔒 **Local embeddings** | Document embeddings always run on-device — only the chat turn goes to Groq |
| 🖥️ **Offline mode** | Set `LLM_PROVIDER=ollama` for a fully local, zero-API-key setup |

---

## 🏗️ Architecture

```
Upload (PDF · DOCX · TXT)
        │
        ▼
Chunk   RecursiveCharacterTextSplitter
        │
        ▼
Embed   Local sentence-transformers model  ← always on-device, never sent to any API
        │
        ▼
Index   FAISS vector store
        │
        ▼
Question ──► Retrieve top-k chunks  (MMR or similarity search)
        │
        ▼
Generate   Groq API → llama / gemma  (or local Ollama if LLM_PROVIDER=ollama)
        │
        ▼
Answer + cited source documents
```

> **Design note:** embeddings and the chat model are intentionally decoupled. Switching models or providers mid-session never invalidates the vector store — only the LLM and conversation chain are rebuilt.

---

## 🗂️ Project Layout

```
smart-rag-chatbot/
├── app.py              Entry point for Streamlit Cloud
├── Main.py             Bootstrap, provider gates, module wiring
├── Session.py          Single source of truth for st.session_state
├── Sidebar.py          Branding, model picker, uploads, stats, controls
├── Chat.py             Conversation rendering + prompt handling
├── Rag.py              The RAG pipeline — chunk, embed, index, retrieve
├── Config.py           Env-driven, validated app configuration
├── design/
│   ├── components.py   Design system (cards, alerts, badges, tables…)
│   └── theme.py        Color tokens / theming
├── static/
│   └── style.css       Chrome styling
└── requirements.txt
```

| File | Responsibility |
|---|---|
| `Main.py` | Bootstraps the app, enforces the active provider's readiness gates, wires everything together |
| `Session.py` | All `st.session_state` reads/writes go through here — nothing else touches raw session keys |
| `Sidebar.py` | Model selection, document upload, stats, session controls |
| `Chat.py` | Renders chat history, handles a single conversational turn end-to-end |
| `Rag.py` | Chunking, embedding, FAISS indexing, retrieval chain, inference |
| `Config.py` | Single source of truth for configuration — env-driven, validated at startup |

---

## ⚙️ Setup

### 1 · Get a free Groq API key

Sign up at [console.groq.com/keys](https://console.groq.com/keys) — takes under a minute, no credit card required.

### 2 · Install dependencies

```bash
git clone https://github.com/Junaid-Ahmed-Rupok/smart-rag-chatbot.git
cd smart-rag-chatbot

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3 · Configure

```bash
cp .env.example .env
```

Open `.env` and paste your key:

```env
GROQ_API_KEY=gsk_your_key_here
```

### 4 · Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) — you're live.

The first question you ask will download the local embedding model (~80 MB, one-time only).

---

### 🖥️ Offline / Local Mode (optional)

To run 100% locally with no API key:

```bash
# Install Ollama from https://ollama.com, then:
ollama serve
ollama pull mistral     # or: llama3 · phi3 · gemma:2b · codellama
```

Then in `.env`:

```env
LLM_PROVIDER=ollama
```

---

## ☁️ Deploying to Streamlit Cloud

1. Push your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo · Branch: `main` · Main file: `app.py`
4. Open **Advanced settings → Secrets** and add:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

5. Click **Deploy** — you'll get a public URL in ~60 seconds.

---

## 🔧 Configuration Reference

All settings are environment-driven via `.env` and validated at startup in `Config.py`.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` (hosted) or `ollama` (fully local) |
| `GROQ_API_KEY` | — | **Required** when `LLM_PROVIDER=groq`. [Get one free](https://console.groq.com/keys) |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Default Groq chat model |
| `LOCAL_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Always local, regardless of provider |
| `DEFAULT_MODEL` | `mistral` | Default model when `LLM_PROVIDER=ollama` |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL (only used when `LLM_PROVIDER=ollama`) |
| `TEMPERATURE` | `0.2` | LLM sampling temperature (`0`–`2`) |
| `TOP_P` | `0.95` | Nucleus sampling parameter |
| `REPEAT_PENALTY` | `1.1` | Penalizes repeated tokens (Ollama only) |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `RETRIEVAL_K` | `4` | Chunks retrieved per question |
| `RETRIEVAL_FETCH_K` | `8` | Candidate pool size before MMR re-ranking |
| `SEARCH_TYPE` | `mmr` | `mmr` (diverse results) or `similarity` |
| `MMR_LAMBDA` | `0.5` | Relevance / diversity trade-off for MMR (`0`–`1`) |
| `CONVERSATION_MEMORY_LENGTH` | `10` | Number of past turns retained in memory |
| `MAX_UPLOAD_SIZE_MB` | `200` | Max upload size per file |
| `DEBUG` | `False` | Verbose logging + LangChain chain trace |

---

## 💬 Usage

1. Open the app — the sidebar shows **● Groq connected** once your key is verified.
2. Pick a model from the **Model** dropdown.
3. Upload one or more PDF / DOCX / TXT files under **Documents**, then click **Index documents**.
4. Ask a question in the chat box — answers cite the source document(s) used.
5. No documents indexed yet? Just ask anyway — it falls back to the bare chat model.

**Sidebar controls:**
- **Clear chat** — wipes conversation history, keeps indexed documents intact
- **Reset all** — wipes everything including the vector store

---

## 🛣️ Roadmap

- [ ] Persistent vector store across sessions (FAISS save/load to `VECTOR_STORE_PATH`)
- [ ] Streaming token-by-token responses
- [ ] Per-document deletion from the index
- [ ] Docker / `docker-compose` setup
- [ ] Automated test suite

Contributions and suggestions welcome — open an issue or a PR.

---

## 👨‍💻 About the Developer

<div align="center">

<img src="https://avatars.githubusercontent.com/Junaid-Ahmed-Rupok" width="100" style="border-radius:50%"/>

### Sarder Junaid Ahmed
**Data Scientist & Machine Learning Engineer**

*Transforming complex data into strategic decisions through rigorous statistical modeling and production-ready machine learning systems.*

[![GitHub](https://img.shields.io/badge/GitHub-Junaid--Ahmed--Rupok-181717?logo=github)](https://github.com/Junaid-Ahmed-Rupok)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sarder%20Junaid%20Ahmed-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sarder-junaid-ahmed-059b68240/)
[![Portfolio](https://img.shields.io/badge/Portfolio-junaid--ahmed--rupok.github.io-1E88E5?logo=githubpages&logoColor=white)](https://junaid-ahmed-rupok.github.io/__portfolio__Yes/)
[![Email](https://img.shields.io/badge/Email-junaidahmedrupok%40gmail.com-EA4335?logo=gmail&logoColor=white)](mailto:junaidahmedrupok@gmail.com)

</div>

**Specializations:** Statistical ML · Causal Inference · Trustworthy AI · Fairness-Aware ML · RAG Systems

**Selected Research:**

- 📄 **Ahmed, S.J.** et al. (2026). *Machine Learning for Crime Classification: A Fairness-Aware Approach to Class Imbalance.* Journal of Machine Learning and Applications, 2(1), 9–17. [DOI: 10.61577/jmla.2026.100002](https://doi.org/10.61577/jmla.2026.100002)
- 📄 **Ahmed, S.J.** et al. (2026). *CF-EGAT: A Causal Fairness-Aware Equity Graph Attention Network for Country-Level Environmental Livability Classification.* SPECTRA 2026. 🏆 **1st Best Paper Award**
- 📄 **Ahmed, S.J.** (2025). *Multi-Dimensional Statistical Similarity for Governance Classification: Beyond Arbitrary Thresholds.* APMEE 2025. 🏆 **Best Research Paper Award**

**Other Deployed Projects:**

- 🔬 [ReproHub](https://reproapp-8jb7vbhnqyltxq23bsr8xn.streamlit.app/) — Automated research reproducibility platform with composite scoring across 11 statistical tests
- 📊 [StatsPro](https://statistical-analysis-app-7axetqtx75ncuu7fr8irxj.streamlit.app/) — AI-powered statistical analysis platform with automated CSV-to-report workflows

**Honors:**
🏆 1st Best Paper — SPECTRA 2026 &nbsp;·&nbsp;
🏆 Best Research Paper — APMEE 2025 &nbsp;·&nbsp;
🎖️ Esteemed Alumni Award — YLRL RUET 2024 &nbsp;·&nbsp;
⭐ Perfect GPA 5.00/5.00 — SSC & HSC &nbsp;·&nbsp;
🎓 National Merit Scholarship — 2009 & 2013

---

## 📄 License

MIT — see [LICENSE](LICENSE).

<div align="center">

Built with [Streamlit](https://streamlit.io) · [LangChain](https://www.langchain.com) · [Groq](https://groq.com) · [FAISS](https://github.com/facebookresearch/faiss) · [sentence-transformers](https://www.sbert.net)

</div>
```
