"""
Central Configuration for Agentic AI Practicals
"""
import os

# Default Ollama settings
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
