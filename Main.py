"""
Smart RAG Chatbot - Main Application
FREE Local LLM Version with Ollama - No API Key Required!
Senior Engineer: Production-grade RAG with 100% local processing
"""

import streamlit as st
from datetime import datetime
from typing import List, Dict, Any

from Config import APP_NAME, APP_ICON, DEBUG
from Rag import get_rag_pipeline, init_rag
from design.components import apply_professional_theme, ds

apply_professional_theme()
init_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "ollama_connected" not in st.session_state:
    st.session_state.ollama_connected = False

def check_ollama():
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                model_names = [m.get("name", "").split(":")[0] for m in models]
                return True, model_names
            return True, ["No models found"]
        return False, []
    except:
        return False, []

def get_installed_models():
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [m.get("name", "").split(":")[0] for m in models]
    except:
        pass
    return []

ds.header(APP_NAME, "FREE Local LLM - No API Key Required! Powered by Ollama", APP_ICON)

with st.sidebar:
    ds.header("⚙️ Configuration", "", "⚙️")
    
    with st.expander("🆓 Local LLM Settings (FREE)", expanded=True):
        st.markdown("✨ 100% FREE - No API Key • No Credit Card • No Internet Required")
        
        ollama_ok, _ = check_ollama()
        installed_models = get_installed_models()
        
        if ollama_ok and installed_models:
            st.session_state.ollama_connected = True
            st.success(f"✅ Ollama connected! ({len(installed_models)} model(s) found)")
            model_name = st.selectbox("Select Model", installed_models)
        else:
            st.session_state.ollama_connected = False
            st.error("❌ Ollama not running!")
            st.markdown("**Install and start Ollama:**")
            st.code("""
# 1. Download from https://ollama.ai
# 2. Install the application
# 3. Open terminal and run:
ollama serve
            
# 4. In another terminal, download a model:
ollama pull mistral
            """, language="bash")
            model_name = "mistral"
    
    ds.divider()
    
    with st.expander("📄 Document Management", expanded=True):
        st.markdown("Upload PDF, DOCX, or TXT files")
        
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
                    if not st.session_state.ollama_connected:
                        st.error("❌ Please start Ollama first")
                    else:
                        try:
                            pipeline = get_rag_pipeline(model_name)
                            with st.spinner(f"Processing {len(new_files)} documents locally..."):
                                num_chunks = pipeline.process_documents(new_files)
                                for f in new_files:
                                    st.session_state.processed_files.add(f.name)
                            st.success(f"✅ Processed {len(new_files)} files into {num_chunks} chunks!")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
            
            if st.session_state.processed_files:
                st.markdown("---")
                st.markdown("**📚 Indexed Documents:**")
                for file in st.session_state.processed_files:
                    st.markdown(f"   • {file}")
    
    ds.divider()
    
    st.markdown("### 📊 System Status")
    
    col1, col2 = st.columns(2)
    with col1:
        doc_count = len(st.session_state.processed_files)
        ds.metric_card("Documents", str(doc_count), "📄", "indexed")
    with col2:
        messages_count = len([m for m in st.session_state.messages if m["role"] == "user"])
        ds.metric_card("Conversations", str(messages_count), "💬", "total")
    
    if st.session_state.ollama_connected:
        st.markdown(f"""
            <div style="background: #D1FAE5; border-radius: 0.5rem; padding: 0.5rem; text-align: center;">
                <span style="color: #065F46;">🟢 Model: {model_name}</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background: #FEE2E2; border-radius: 0.5rem; padding: 0.5rem; text-align: center;">
                <span style="color: #991B1B;">🔴 Ollama: Offline</span>
            </div>
        """, unsafe_allow_html=True)
    
    ds.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.messages = []
            st.session_state.processed_files = set()
            st.session_state.rag_pipeline = None
            st.rerun()
    
    st.markdown("---")
    st.caption("⚡ Powered by Ollama & RAG")
    st.caption("🆓 100% FREE - No API Key")

if st.session_state.messages:
    chat_stats = f"💬 {len(st.session_state.messages)} messages"
    if st.session_state.processed_files:
        chat_stats += f" • 📚 {len(st.session_state.processed_files)} documents"
    if st.session_state.ollama_connected:
        chat_stats += f" • 🟢 {model_name}"
    st.caption(chat_stats)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("📎 Sources", expanded=False):
                for src in message["sources"]:
                    ds.source_card(src)
        if "timestamp" in message:
            st.caption(f"_{message['timestamp']}_")

if not st.session_state.ollama_connected:
    st.warning("""
    🔴 **Ollama is not running!**
    
    Please:
    1. Open terminal and run: `ollama serve`
    2. Download a model: `ollama pull mistral`
    3. Refresh this page
    """)
    st.stop()

docs_loaded = len(st.session_state.processed_files) > 0
if not docs_loaded:
    st.info("📄 **Upload and process documents in the sidebar to ask questions about them**")

placeholder = "Ask me anything about your documents..." if docs_loaded else "Ask me a general question..."
prompt = st.chat_input(placeholder)

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().strftime("%I:%M %p")
    })
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            pipeline = get_rag_pipeline(model_name)
            
            if uploaded_files:
                new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
                if new_files:
                    with st.spinner(f"Processing {len(new_files)} documents..."):
                        pipeline.process_documents(new_files)
                        for f in new_files:
                            st.session_state.processed_files.add(f.name)
            
            with st.spinner(f"🤔 Thinking with {model_name}..."):
                if docs_loaded:
                    result = pipeline.ask(prompt)
                    answer = result["answer"]
                    sources = result["sources"]
                    st.markdown(answer)
                    if sources:
                        with st.expander("📎 Sources", expanded=False):
                            for src in sources:
                                ds.source_card(src)
                else:
                    pipeline.init_models()
                    answer = pipeline.llm.invoke(prompt)
                    sources = []
                    st.markdown(answer)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "timestamp": datetime.now().strftime("%I:%M %p")
            })
            
        except Exception as e:
            error_msg = f"❌ **Error:** {str(e)}"
            if "connection" in str(e).lower():
                error_msg += "\n\nMake sure Ollama is running: `ollama serve`"
            elif "model" in str(e).lower():
                error_msg += f"\n\nModel '{model_name}' not found. Run: `ollama pull {model_name}`"
            
            st.error(error_msg)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "sources": [],
                "timestamp": datetime.now().strftime("%I:%M %p")
            })

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🎯 **Features**")
    st.caption("• Document Q&A (PDF, DOCX, TXT)")
    st.caption("• Source attribution")
    st.caption("• Conversation memory")
    st.caption("• 100% Local & Private")

with col2:
    st.caption("⚡ **Tech Stack**")
    st.caption("• Streamlit + LangChain")
    st.caption("• Ollama + Mistral/Llama")
    st.caption("• FAISS Vector Database")
    st.caption("• No Cloud Dependencies")

with col3:
    st.caption("📊 **Status**")
    if st.session_state.ollama_connected:
        st.caption(f"✅ {model_name}: Ready")
    else:
        st.caption("❌ Ollama: Offline")
    st.caption(f"📚 Documents: {len(st.session_state.processed_files)}")
    st.caption(f"💬 Messages: {len(st.session_state.messages)}")
    st.caption("🆓 100% FREE")

st.markdown(
    """
    <div style="text-align: center; padding: 1rem; color: #64748B; font-size: 0.75rem;">
        🆓 Completely FREE • No API Key • No Credit Card • Runs 100% Locally • Enterprise Ready
    </div>
    """,
    unsafe_allow_html=True
)
