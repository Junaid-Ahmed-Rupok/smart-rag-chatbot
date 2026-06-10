"""
RAG Pipeline - Production Grade Implementation
Senior Engineer: Complete RAG system
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
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

from Config import CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_K

class RAGPipeline:
    """Complete RAG implementation"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.vector_store = None
        self.chain = None
        self.processed_files = set()
        
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
            # Skip already processed files
            if file.name in self.processed_files:
                continue
                
            # Get file extension
            suffix = Path(file.name).suffix.lower()
            if suffix not in loaders:
                st.warning(f"Skipping {file.name} - unsupported format")
                continue
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            
            try:
                # Load document
                loader = loaders[suffix](tmp_path)
                docs = loader.load()
                
                # Add source metadata
                for doc in docs:
                    doc.metadata["source"] = file.name
                
                # Split into chunks
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
                self.processed_files.add(file.name)
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        
        if not all_chunks:
            raise ValueError("No valid documents processed")
        
        # Create or update vector store
        embeddings = OpenAIEmbeddings(api_key=self.api_key)
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(all_chunks, embeddings)
        else:
            self.vector_store.add_documents(all_chunks)
        
        # Reset chain (will be rebuilt with new retriever)
        self.chain = None
        
        return len(all_chunks)
    
    def get_chain(self):
        """Get or create conversation chain"""
        if self.chain is not None:
            return self.chain
        
        if self.vector_store is None:
            raise ValueError("No documents loaded. Upload documents first.")
        
        # Create retriever with MMR search
        retriever = self.vector_store.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": RETRIEVAL_K,
                "fetch_k": RETRIEVAL_K * 2
            }
        )
        
        # Create LLM
        llm = ChatOpenAI(
            model=self.model,
            temperature=0.2,
            api_key=self.api_key
        )
        
        # Create memory for conversation
        memory = ConversationBufferWindowMemory(
            k=5,  # Remember last 5 exchanges
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
        
        return self.chain
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question and get answer with sources"""
        chain = self.get_chain()
        result = chain.invoke({"question": question})
        
        # Extract sources
        sources = []
        for doc in result.get("source_documents", []):
            source = doc.metadata.get("source", "unknown")
            if source not in sources:
                sources.append(source)
        
        return {
            "answer": result["answer"],
            "sources": sources,
            "chat_history": result.get("chat_history", [])
        }
    
    def clear(self):
        """Clear all data"""
        self.vector_store = None
        self.chain = None
        self.processed_files = set()

# Initialize session state for RAG
def init_rag():
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

def get_rag_pipeline(api_key: str, model: str) -> RAGPipeline:
    """Get or create RAG pipeline instance"""
    if st.session_state.rag_pipeline is None:
        st.session_state.rag_pipeline = RAGPipeline(api_key, model)
    return st.session_state.rag_pipeline
