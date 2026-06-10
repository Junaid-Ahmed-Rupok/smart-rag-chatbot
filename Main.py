"""
Smart RAG Chatbot - Main Application
Senior Engineer: Production-grade RAG implementation with professional design
"""

import streamlit as st
import os
from datetime import datetime
from typing import List, Dict, Any

# Import configuration
from Config import APP_NAME, APP_ICON, DEBUG

# Import RAG pipeline
from Rag import get_rag_pipeline, init_rag, RAGPipeline

# Import design components
from design.components import apply_professional_theme, ds, chat_bubble

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

apply_professional_theme()

# Initialize session state
init_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# ============================================================================
# HEADER SECTION
# ============================================================================

ds.header(
    title=APP_NAME,
    subtitle="Enterprise-grade Retrieval-Augmented Generation Chatbot with Professional Design System",
    icon=APP_ICON
)

# ============================================================================
# SIDEBAR SECTION
# ============================================================================

with st.sidebar:
    ds.header("⚙️ Configuration", "", "⚙️")
    
    # API Settings
    with st.expander("🔑 API Settings", expanded=True):
        api_key = st.text_input(
            "OpenAI API Key", 
            type="password", 
            placeholder="sk-...",
            help="Enter your OpenAI API key. Get one from platform.openai.com"
        )
        
        model = st.selectbox(
            "Select Model",
            ["gpt-4", "gpt-3.5-turbo"],
            index=1,
            help="GPT-4 is more accurate but slower. GPT-3.5-turbo is faster and cheaper"
        )
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            ds.status_badge("API Connected", "success")
        else:
            ds.status_badge("API Required", "warning")
    
    ds.divider()
    
    # Document Upload Section
    with st.expander("📄 Document Management", expanded=True):
        st.markdown("""
            <div style="font-size: 0.875rem; color: #64748B; margin-bottom: 0.5rem;">
                Upload PDF, DOCX, or TXT files
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
            
            if new_files:
                ds.status_badge(f"{len(new_files)} new file(s) ready", "info")
                
                if st.button("📚 Process Documents", use_container_width=True, type="primary"):
                    if not api_key:
                        st.error("❌ Please enter your OpenAI API key first")
                    else:
                        try:
                            pipeline = get_rag_pipeline(api_key, model)
                            with st.spinner("Processing documents..."):
                                num_chunks = pipeline.process_documents(new_files)
                                for f in new_files:
                                    st.session_state.processed_files.add(f.name)
                            st.success(f"✅ Processed {len(new_files)} files into {num_chunks} chunks!")
                            ds.toast("Documents processed successfully", "success")
                        except Exception as e:
                            st.error(f"❌ Error processing documents: {e}")
            
            # Show processed files
            if st.session_state.processed_files:
                st.markdown("---")
                st.markdown("**📚 Indexed Documents:**")
                for file in st.session_state.processed_files:
                    st.markdown(f"   • {file}")
    
    ds.divider()
    
    # System Status
    st.markdown("### 📊 System Status")
    
    col1, col2 = st.columns(2)
    with col1:
        ds.metric_card(
            "Documents", 
            str(len(st.session_state.processed_files)), 
            "📄",
            "indexed"
        )
    with col2:
        messages_count = len([m for m in st.session_state.messages if m["role"] == "user"])
        ds.metric_card("Conversations", str(messages_count), "💬", "total")
    
    ds.divider()
    
    # Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.rag_pipeline:
                st.session_state.rag_pipeline.chain = None
            ds.toast("Chat history cleared", "success")
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.messages = []
            st.session_state.processed_files = set()
            st.session_state.rag_pipeline = None
            ds.toast("All data reset", "info")
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.caption("⚡ Powered by RAG Architecture")
    st.caption("🎨 Professional Design System v2.0")
    st.caption("🔒 Enterprise Security Ready")

# ============================================================================
# MAIN CHAT AREA
# ============================================================================

# Chat header with stats
if st.session_state.messages:
    chat_stats = f"💬 {len(st.session_state.messages)} messages"
    if st.session_state.processed_files:
        chat_stats += f" • 📚 {len(st.session_state.processed_files)} documents"
    st.caption(chat_stats)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("📎 Sources", expanded=False):
                for src in message["sources"]:
                    ds.source_card(src)
        
        # Show timestamp
        if "timestamp" in message:
            st.caption(f"_{message['timestamp']}_")

# ============================================================================
# CHAT INPUT SECTION
# ============================================================================

# Check if API key is provided
if not api_key:
    st.info("🔑 **Please enter your OpenAI API key in the sidebar to start chatting**")
    st.stop()

# Check if documents are loaded
docs_loaded = len(st.session_state.processed_files) > 0
if not docs_loaded:
    st.info("📄 **Upload and process documents in the sidebar to ask questions about them**")
    st.warning("⚠️ Without documents, I can only answer general questions (no RAG context)")

# Chat input
placeholder = "Ask me anything about your documents..." if docs_loaded else "Ask me a general question... (upload documents for RAG context)"
prompt = st.chat_input(placeholder)

if prompt:
    # Add user message to history
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt,
        "timestamp": datetime.now().strftime("%I:%M %p")
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        try:
            # Get RAG pipeline
            pipeline = get_rag_pipeline(api_key, model)
            
            # Process any pending documents
            if uploaded_files:
                new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
                if new_files:
                    with st.spinner("Processing documents..."):
                        pipeline.process_documents(new_files)
                        for f in new_files:
                            st.session_state.processed_files.add(f.name)
            
            # Generate response
            with st.spinner("🤔 Thinking and retrieving relevant information..."):
                if docs_loaded:
                    # RAG-enabled response
                    result = pipeline.ask(prompt)
                    answer = result["answer"]
                    sources = result["sources"]
                    
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📎 Sources", expanded=False):
                            for src in sources:
                                ds.source_card(src)
                else:
                    # General response without RAG
                    from langchain_openai import ChatOpenAI
                    from langchain.chains import ConversationChain
                    from langchain.memory import ConversationBufferMemory
                    
                    llm = ChatOpenAI(model=model, temperature=0.7, api_key=api_key)
                    memory = ConversationBufferMemory()
                    chain = ConversationChain(llm=llm, memory=memory)
                    answer = chain.predict(input=prompt)
                    
                    st.markdown(answer)
                    sources = []
            
            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "timestamp": datetime.now().strftime("%I:%M %p")
            })
            
        except Exception as e:
            error_msg = f"❌ **Error:** {str(e)}"
            if "api_key" in str(e).lower():
                error_msg += "\n\nPlease check your OpenAI API key in the sidebar."
            elif "rate limit" in str(e).lower():
                error_msg += "\n\nYou've hit the rate limit. Please wait a moment and try again."
            
            st.error(error_msg)
            
            # Add error message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "sources": [],
                "timestamp": datetime.now().strftime("%I:%M %p")
            })

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🎯 **Features**")
    st.caption("• Document Q&A (PDF, DOCX, TXT)")
    st.caption("• Source attribution")
    st.caption("• Conversation memory")

with col2:
    st.caption("⚡ **Tech Stack**")
    st.caption("• Streamlit + LangChain")
    st.caption("• OpenAI GPT + Embeddings")
    st.caption("• FAISS Vector Database")

with col3:
    st.caption("📊 **Status**")
    if api_key:
        st.caption("✅ API: Connected")
    else:
        st.caption("⚠️ API: Not connected")
    st.caption(f"📚 Documents: {len(st.session_state.processed_files)}")
    st.caption(f"💬 Messages: {len(st.session_state.messages)}")

st.markdown(
    """
    <div style="text-align: center; padding: 1rem; color: #64748B; font-size: 0.75rem;">
        Built with ❤️ using Professional Design System | Enterprise Ready | Production Grade
    </div>
    """, 
    unsafe_allow_html=True
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_health():
    """Health check for monitoring"""
    return {
        "status": "healthy",
        "documents_loaded": len(st.session_state.processed_files),
        "messages_count": len(st.session_state.messages),
        "api_configured": bool(api_key)
    }
