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

from design.components import apply_professional_theme
from Rag import init_rag, get_rag_pipeline  # Corrected import
from Config import APP_ICON, APP_NAME, APP_VERSION, bootstrap, cfg

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

        # ── PROCESS FILES LOGIC ──────────────────────────────────────────────
        if uploaded_files:
            # Get the current pipeline or create one
            pipeline = get_rag_pipeline(cfg.groq_model)
            
            # Track new files to avoid re-processing
            new_files = []
            for file in uploaded_files:
                # Check if we've already processed this file by name
                if file.name not in st.session_state.processed_files:
                    new_files.append(file)
            
            if new_files:
                with st.spinner(f"Processing {len(new_files)} document(s)..."):
                    try:
                        # Process the documents using the pipeline's method
                        pipeline.process_documents(new_files)
                        
                        # Add file names to session state so we don't process them again
                        for file in new_files:
                            st.session_state.processed_files.add(file.name)
                        
                        # Store the pipeline back into session state
                        st.session_state.rag_pipeline = pipeline
                        
                        # Refresh the page to update the Docs counter
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")
        # ──────────────────────────────────────────────────────────────────────

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
                
                # ── ACTUAL RAG PIPELINE LOGIC ──
                if st.session_state.rag_pipeline:
                    try:
                        # Use the pipeline's ask() method
                        result = st.session_state.rag_pipeline.ask(prompt)
                        response = result["answer"]
                    except Exception as e:
                        response = f"An error occurred while processing your question: {str(e)}"
                else:
                    # Fallback if pipeline isn't ready (e.g., no documents uploaded yet)
                    response = "Please upload a document first so I can answer your question based on its content."
                # ──────────────────────────────────
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
