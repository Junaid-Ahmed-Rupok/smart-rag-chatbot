import streamlit as st
import os

# ── FORCE LOAD STREAMLIT SECRETS INTO ENVIRONMENT ──────────────────────────
# This must happen BEFORE any other imports that rely on the API key.

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    print("✅ Successfully loaded GROQ_API_KEY from Streamlit Secrets.")
else:
    print("❌ WARNING: GROQ_API_KEY not found in Streamlit Secrets!")

# ── IMPORT REST OF APP ──────────────────────────────────────────────────────
# Now import everything else as normal

from design.components import apply_professional_theme, ds
from Rag import init_rag
from Config import APP_ICON, APP_NAME, APP_VERSION, bootstrap, cfg  # <-- ADDED APP_VERSION HERE
import Session

# ── Session state ────────────────────────────────────────────────────────────

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Bootstrap directories
    bootstrap(cfg)

    # Apply custom theme
    apply_professional_theme()

    # Initialize RAG
    init_rag()

    # Initialize session state
    init_session_state()

    # ── Header ────────────────────────────────────────────────────────────────

    st.markdown(f"# {APP_ICON} {APP_NAME}")
    st.caption(f"v{APP_VERSION} · powered by Groq")

    # ── Sidebar ──────────────────────────────────────────────────────────────

    with st.sidebar:
        st.markdown("### Model")
        if cfg.llm_provider == "groq":
            model_name = cfg.groq_model
            st.success(f"● {model_name}")
        else:
            model_name = cfg.default_model
            st.info(f"● {model_name} (Ollama)")

        st.markdown("---")
        st.markdown("### Troubleshooting")
        with st.expander("❓ Help"):
            st.markdown("""
            **LLM Provider:** Groq (hosted)  
            **Embeddings:** Local (sentence-transformers)  
            **Chunk Size:** {}  
            **Retrieval K:** {}  
            """.format(cfg.chunk_size, cfg.retrieval_k))

        st.markdown("---")
        st.markdown("### Documents")

        uploaded_files = st.file_uploader(
            "Upload",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if uploaded_files:
            # This part handles file processing logic (you can adapt this based on your Rag.py)
            pass

        st.caption(f"{cfg.max_upload_mb}MB per file • PDF, DOCX, TXT")

        # Stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Docs", len(st.session_state.processed_files))
        with col2:
            st.metric("💬 Turns", len(st.session_state.messages))

    # ── Chat Interface ────────────────────────────────────────────────────────

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask something about your documents..."):
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Placeholder response - you'll connect this to your RAG pipeline
                response = "I'm ready to answer! I just need to be connected to the RAG pipeline implementation."
                
                # If you have a RAG chain:
                # if st.session_state.rag_pipeline:
                #     response = st.session_state.rag_pipeline.invoke(prompt)
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
