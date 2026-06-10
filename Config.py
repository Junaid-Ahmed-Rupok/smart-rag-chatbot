"""
Smart RAG Chatbot - Configuration File
Senior Engineer: Centralized configuration management with validation
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# APP CONFIGURATION
# ============================================================================

APP_NAME = "Smart RAG Chatbot"
APP_ICON = "🤖"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "Enterprise-grade Retrieval-Augmented Generation Chatbot"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# ============================================================================
# API CONFIGURATION
# ============================================================================

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", "")

# Model Settings
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
AVAILABLE_MODELS: List[str] = [
    "gpt-4",
    "gpt-4-turbo-preview", 
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k"
]

# Embedding Settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
EMBEDDING_DIMENSIONS = 1536  # ada-002 dimensions

# LLM Parameters
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
FREQUENCY_PENALTY = float(os.getenv("FREQUENCY_PENALTY", "0.0"))
PRESENCE_PENALTY = float(os.getenv("PRESENCE_PENALTY", "0.0"))

# ============================================================================
# RAG CONFIGURATION
# ============================================================================

# Text Splitting
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
CHUNK_SEPARATORS = ["\n\n", "\n", " ", ""]

# Retrieval Settings
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
RETRIEVAL_FETCH_K = int(os.getenv("RETRIEVAL_FETCH_K", "8"))
SEARCH_TYPE = os.getenv("SEARCH_TYPE", "mmr")  # "similarity" or "mmr"
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))  # 0 = max diversity, 1 = max similarity

# Memory Settings
CONVERSATION_MEMORY_LENGTH = int(os.getenv("CONVERSATION_MEMORY_LENGTH", "10"))
MEMORY_KEY = "chat_history"
OUTPUT_KEY = "answer"

# ============================================================================
# FILE CONFIGURATION
# ============================================================================

# Supported File Types
SUPPORTED_FILE_TYPES: Dict[str, List[str]] = {
    "document": [".pdf", ".docx", ".txt"],
    "image": [".png", ".jpg", ".jpeg"],
    "spreadsheet": [".csv", ".xlsx"]
}
SUPPORTED_EXTENSIONS = [ext for exts in SUPPORTED_FILE_TYPES.values() for ext in exts]

# File Size Limits (in MB)
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Vector Store Configuration
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss")  # "faiss" or "chroma"
VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", "./data/vector_store"))
VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

# Data Paths
DATA_PATH = Path("./data")
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
LOGS_PATH = Path("./logs")

# Create directories
for path in [DATA_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH, LOGS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# UI CONFIGURATION
# ============================================================================

# Theme Colors (matches CSS design tokens)
THEME = {
    "primary": "#1E88E5",
    "primary_dark": "#0D47A1",
    "primary_light": "#42A5F5",
    "secondary": "#FF6B6B",
    "success": "#10B981",
    "warning": "#F59E0B",
    "error": "#EF4444",
    "info": "#3B82F6",
    "background": "#F5F7FA",
    "surface": "#FFFFFF",
    "text_primary": "#1E293B",
    "text_secondary": "#64748B",
    "border": "#E2E8F0"
}

# Layout Settings
LAYOUT = {
    "sidebar_width": 320,
    "max_content_width": 1200,
    "chat_max_width": 800
}

# Typography
TYPOGRAPHY = {
    "font_family": "'Inter', -apple-system, sans-serif",
    "font_code": "'Fira Code', monospace",
    "font_sizes": {
        "xs": "0.75rem",
        "sm": "0.875rem", 
        "base": "1rem",
        "lg": "1.125rem",
        "xl": "1.25rem",
        "2xl": "1.5rem",
        "3xl": "1.875rem",
        "4xl": "2.25rem"
    }
}

# ============================================================================
# SESSION STATE KEYS
# ============================================================================

SESSION_KEYS = {
    "messages": "chat_messages",
    "rag_pipeline": "rag_pipeline_instance",
    "processed_files": "processed_documents_set",
    "vector_store": "vector_store_instance",
    "uploaded_files": "uploaded_files_list",
    "chain": "conversation_chain"
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG" if DEBUG else "INFO",
            "formatter": "default"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": LOGS_PATH / "app.log",
            "level": "INFO",
            "formatter": "detailed"
        }
    },
    "root": {
        "level": "DEBUG" if DEBUG else "INFO",
        "handlers": ["console", "file"]
    }
}

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_config() -> Dict[str, bool]:
    """Validate configuration settings"""
    validation = {
        "api_key_exists": bool(OPENAI_API_KEY),
        "data_path_writable": DATA_PATH.exists() and os.access(DATA_PATH, os.W_OK),
        "vector_store_writable": VECTOR_STORE_PATH.exists() and os.access(VECTOR_STORE_PATH, os.W_OK),
        "chunk_size_valid": 100 <= CHUNK_SIZE <= 5000,
        "chunk_overlap_valid": 0 <= CHUNK_OVERLAP <= CHUNK_SIZE,
        "retrieval_k_valid": 1 <= RETRIEVAL_K <= 20,
        "temperature_valid": 0 <= TEMPERATURE <= 2,
        "model_valid": DEFAULT_MODEL in AVAILABLE_MODELS
    }
    return validation

def get_config_summary() -> Dict[str, any]:
    """Get configuration summary for display"""
    return {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "debug_mode": DEBUG,
        "model": DEFAULT_MODEL,
        "chunk_size": CHUNK_SIZE,
        "retrieval_k": RETRIEVAL_K,
        "temperature": TEMPERATURE,
        "supported_file_types": SUPPORTED_EXTENSIONS,
        "vector_store_path": str(VECTOR_STORE_PATH),
        "memory_length": CONVERSATION_MEMORY_LENGTH
    }

def is_ready() -> bool:
    """Check if app is ready to run"""
    validations = validate_config()
    return validations["api_key_exists"] and validations["data_path_writable"]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_file_size_limit_mb() -> int:
    """Get max file upload size in MB"""
    return MAX_UPLOAD_SIZE_MB

def get_supported_extensions() -> List[str]:
    """Get list of supported file extensions"""
    return SUPPORTED_EXTENSIONS

def get_chunk_config() -> Dict[str, int]:
    """Get text chunking configuration"""
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP
    }

def get_retrieval_config() -> Dict[str, any]:
    """Get retrieval configuration"""
    return {
        "k": RETRIEVAL_K,
        "fetch_k": RETRIEVAL_FETCH_K,
        "search_type": SEARCH_TYPE,
        "mmr_lambda": MMR_LAMBDA
    }

# ============================================================================
# ENVIRONMENT SPECIFIC CONFIGURATIONS
# ============================================================================

if DEBUG:
    # Development settings
    TEMPERATURE = 0.3  # Slightly more creative for testing
    LOGGING_CONFIG["handlers"]["console"]["level"] = "DEBUG"
else:
    # Production settings
    LOGGING_CONFIG["handlers"]["console"]["level"] = "INFO"

# ============================================================================
# EXPORT CONFIGURATION DICTIONARY
# ============================================================================

CONFIG = {
    "app": {
        "name": APP_NAME,
        "icon": APP_ICON,
        "version": APP_VERSION,
        "description": APP_DESCRIPTION,
        "debug": DEBUG
    },
    "api": {
        "openai_key": OPENAI_API_KEY,
        "default_model": DEFAULT_MODEL,
        "available_models": AVAILABLE_MODELS,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    },
    "rag": {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "retrieval_k": RETRIEVAL_K,
        "search_type": SEARCH_TYPE,
        "memory_length": CONVERSATION_MEMORY_LENGTH
    },
    "files": {
        "supported_extensions": SUPPORTED_EXTENSIONS,
        "max_upload_size_mb": MAX_UPLOAD_SIZE_MB,
        "data_path": str(DATA_PATH)
    },
    "ui": {
        "theme": THEME,
        "layout": LAYOUT,
        "typography": TYPOGRAPHY
    }
}

# ============================================================================
# PRINT CONFIG STATUS (only in debug mode)
# ============================================================================

if DEBUG:
    print("=" * 60)
    print(f"📋 {APP_NAME} v{APP_VERSION} Configuration Loaded")
    print("=" * 60)
    print(f"✅ Model: {DEFAULT_MODEL}")
    print(f"✅ Chunk Size: {CHUNK_SIZE}")
    print(f"✅ Retrieval K: {RETRIEVAL_K}")
    print(f"✅ Data Path: {DATA_PATH}")
    print(f"✅ Vector Store: {VECTOR_STORE_PATH}")
    print(f"✅ Supported Files: {', '.join(SUPPORTED_EXTENSIONS)}")
    
    validations = validate_config()
    print("\n📊 Validation Results:")
    for key, value in validations.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key}: {value}")
    print("=" * 60)
