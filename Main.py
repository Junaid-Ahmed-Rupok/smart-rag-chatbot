"""
Smart RAG Chatbot - Main Application
Senior Engineer: Production-grade implementation
"""

import streamlit as st
from Config import APP_NAME, APP_ICON, DEBUG
from design.components import apply_professional_theme, ds

# Apply professional theme
apply_professional_theme()

# Header with design system
ds.header(
    title=APP_NAME,
    subtitle="Enterprise-grade Retrieval-Augmented Generation Chatbot",
    icon=APP_ICON
)

# Sidebar with professional components
with st.sidebar:
    ds.header("⚙️ Configuration", "", "⚙️")
    
    # API Configuration section
    with st.expander("🔑 API Settings", expanded=True):
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        model = st.selectbox("Model", ["gpt-4", "gpt-3.5-turbo"])
    
    ds.divider()
    
    # Document upload section
    with st.expander("📄 Document Management", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        if uploaded_files:
            ds.status_badge(f"{len(uploaded_files)} files uploaded", "success")
    
    ds.divider()
    
    # Status section
    ds.metric_card("System Status", "Online", "🟢", "All systems operational")
    
    # Clear chat button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        ds.toast("Conversation cleared", "success")
        st.rerun()

# Main chat area
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📎 Sources"):
                for src in message["sources"]:
                    ds.source_card(src)

# Chat input
if prompt := st.chat_input("Ask me anything about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response (placeholder - integrate Rag.py here)
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            # TODO: Integrate RAG pipeline from Rag.py
            response = "✅ RAG pipeline ready! Connect your document processing and LLM logic here."
            st.markdown(response)
            
            # Show source documents if available
            # with st.expander("📎 Sources"):
            #     ds.source_card("sample.pdf", "Preview of content...")
            
            st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("""
    <div style="
        text-align: center;
        padding: 1rem;
        color: #64748B;
        font-size: 0.75rem;
        border-top: 1px solid #E2E8F0;
        margin-top: 2rem;
    ">
        ⚡ Powered by RAG Architecture | 🎨 Professional Design System | 🔒 Enterprise Security
    </div>
""", unsafe_allow_html=True)
