"""
RAG Pipeline - FREE Local LLM Version (Ollama)
No API key needed - Runs entirely on your laptop
"""

import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

from Config import CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_K

class RAGPipeline:
    """Complete RAG implementation - FREE with Ollama"""
    
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
        self.vector_store = None
        self.chain = None
        self.processed_files = set()
        self.embeddings = None
        self.llm = None
        
    def check_ollama(self) -> bool:
        """Check if Ollama is running"""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            return response.status_code == 200
        except:
            return False
    
    def init_models(self):
        """Initialize Ollama models"""
        if self.embeddings is None:
            self.embeddings = OllamaEmbeddings(model=self.model_name)
        if self.llm is None:
            self.llm = Ollama(model=self.model_name, temperature=0.2)
        return True
        
    def process_documents(self, files: List) -> int:
        """Process uploaded documents and create vector store"""
        all_chunks = []
        
        # Loaders for different file types
        loaders = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt": TextLoader
        }
        
        # Text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        
        for file in files:
            if file.name in self.processed_files:
                continue
                
            suffix = Path(file.name).suffix.lower()
            if suffix not in loaders:
                st.warning(f"Skipping {file.name}")
                continue
            
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            
            try:
                loader = loaders[suffix](tmp_path)
                docs = loader.load()
                
                for doc in docs:
                    doc.metadata["source"] = file.name
                
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
                self.processed_files.add(file.name)
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        
        if not all_chunks:
            raise ValueError("No valid documents processed")
        
        self.init_models()
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(all_chunks, self.embeddings)
        else:
            self.vector_store.add_documents(all_chunks)
        
        self.chain = None
        return len(all_chunks)
    
    def get_chain(self):
        """Get or create conversation chain"""
        if self.chain is not None:
            return self.chain
        
        if self.vector_store is None:
            raise ValueError("No documents loaded")
        
        self.init_models()
        
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": RETRIEVAL_K, "fetch_k": RETRIEVAL_K * 2}
        )
        
        memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        
        return self.chain
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question and get answer with sources"""
        chain = self.get_chain()
        result = chain.invoke({"question": question})
        
        sources = []
        for doc in result.get("source_documents", []):
            source = doc.metadata.get("source", "unknown")
            if source not in sources:
                sources.append(source)
        
        return {
            "answer": result["answer"],
            "sources": sources
        }
    
    def clear(self):
        """Clear all data"""
        self.vector_store = None
        self.chain = None
        self.processed_files = set()

def init_rag():
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

def get_rag_pipeline(model_name: str = "mistral") -> RAGPipeline:
    if st.session_state.rag_pipeline is None:
        st.session_state.rag_pipeline = RAGPipeline(model_name)
    return st.session_state.rag_pipeline
