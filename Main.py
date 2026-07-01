import streamlit as st
import os

# ── FORCE LOAD STREAMLIT SECRETS INTO ENVIRONMENT ──────────────────────────
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    print("✅ Successfully loaded GROQ_API_KEY from Streamlit Secrets.")
else:
    print("❌ WARNING: GROQ_API_KEY not found in Streamlit Secrets!")

# ── IMPORT REST OF APP ──────────────────────────────────────────────────────
from design.components import apply_professional_theme
from Rag import init_rag, get_rag_pipeline
from Config import APP_ICON, APP_NAME, APP_VERSION, bootstrap, cfg
import Session  # <-- Import Session


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 1. Initialize session state FIRST (creates all default keys)
    Session.init()

    # 2. Bootstrap directories
    bootstrap(cfg)

    # 3. Apply custom theme
    apply_professional_theme()

    # 4. Initialize RAG
    init_rag()

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

        # ── PROCESS FILES LOGIC ──────────────────────────────────────────────
        if uploaded_files:
            pipeline = get_rag_pipeline(cfg.groq_model)
            
            new_files = []
            for file in uploaded_files:
                if file.name not in Session.processed_files():
                    new_files.append(file)
            
            if new_files:
                with st.spinner(f"Processing {len(new_files)} document(s)..."):
                    try:
                        pipeline.process_documents(new_files)
                        Session.mark_processed([f.name for f in new_files])
                        st.session_state.rag_pipeline = pipeline
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")
        # ──────────────────────────────────────────────────────────────────────

        st.caption(f"{cfg.max_upload_mb}MB per file • PDF, DOCX, TXT")

        # Stats - Use Session functions
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Docs", len(Session.processed_files()))
        with col2:
            st.metric("💬 Turns", Session.turn_count())

    # ── Chat Interface ────────────────────────────────────────────────────────

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask something about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.rag_pipeline:
                    try:
                        result = st.session_state.rag_pipeline.ask(prompt)
                        response = result["answer"]
                    except Exception as e:
                        response = f"An error occurred while processing your question: {str(e)}"
                else:
                    response = "Please upload a document first so I can answer your question based on its content."
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
