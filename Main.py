"""
Smart RAG Chatbot - Main Application
FREE Local LLM Version with Ollama - No API Key Required!
Senior Engineer: Production-grade RAG with 100% local processing
"""

import streamlit as st
import subprocess
from datetime import datetime
from typing import List, Dict, Any

# Import configuration
from Config import APP_NAME, APP_ICON, DEBUG

# Import RAG pipeline (FREE version)
from Rag import get_rag_pipeline, init_rag, RAGPipeline

# Import design components
from design.components import apply_professional_theme, ds

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
if "ollama_connected" not in st.session_state:
    st.session_state.ollama_connected = False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_ollama() -> tuple[bool, str]:
    """Check if Ollama is running and available"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                model_names = [m.get("name", "").split(":")[0] for m in models]
                return True, model_names
            else:
                return True, ["No models found. Run 'ollama pull mistral'"]
        return False, []
    except requests.exceptions.ConnectionError:
        return False, []
    except Exception as e:
        return False, [str(e)]

def get_installed_models() -> List[str]:
    """Get list of installed Ollama models"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [m.get("name", "").split(":")[0] for m in models]
    except:
        pass
    return []

# ============================================================================
# HEADER SECTION
# ============================================================================

ds.header(
    title=APP_NAME,
    subtitle="FREE Local LLM - No API Key Required! Powered by Ollama",
    icon=APP_ICON
)

# ============================================================================
# SIDEBAR SECTION
# ============================================================================

with st.sidebar:
    ds.header("⚙️ Configuration", "", "⚙️")
    
    # Ollama Connection Status
    with st.expander("🆓 Local LLM Settings (FREE)", expanded=True):
        st.markdown("""
            <div style="
                background: linear-gradient(135deg, #10B98120, #05966920);
                border-radius: 0.5rem;
                padding: 0.75rem;
                margin-bottom: 1rem;
            ">
                <div style="font-size: 0.875rem; font-weight: 600; color: #10B981;">✨ 100% FREE</div>
                <div style="font-size: 0.75rem; color: #047857;">No API Key • No Credit Card • No Internet Required</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Check Ollama status
        ollama_ok, models_data = check_ollama()
        
        if ollama_ok:
            installed_models = get_installed_models()
            if installed_models:
                st.session_state.ollama_connected = True
                st.success(f"✅ Ollama connected! ({len(installed_models)} model(s) found)")
                
                # Model selection
                model_name = st.selectbox(
                    "Select Model",
                    installed_models,
                    help="Choose a model you've downloaded with Ollama"
                )
            else:
                st.session_state.ollama_connected = False
                st.warning("⚠️ No models found. Install a model:")
                st.code("ollama pull mistral", language="bash")
                model_name = "mistral"
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
        
        st.divider()
        
        # Model info
        with st.expander("ℹ️ About Ollama Models", expanded=False):
            st.markdown("""
            **Recommended Models:**
            - `mistral` - 4GB, very good quality (⭐ recommended)
            - `llama3` - 4.6GB, excellent quality
            - `phi3` - 2.3GB, lightweight & fast
            - `gemma:2b` - 1.6GB, fastest
            
            **Install a model:**
            ```bash
            ollama pull mistral
