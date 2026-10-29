"""
Practical: 14. RAG + Embeddings + FAISS
Objective: Retrieval-Augmented Generation using all-MiniLM-L6-v2 embeddings, FAISS vector store, and LLaMA 3.2.
"""
import sys
import os

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

KNOWLEDGE_BASE = [
    "Agentic AI uses iterative goal decomposition and tool use.",
    "FAISS is a library for dense vector similarity search developed by Meta.",
    "Ollama runs LLMs locally on consumer hardware."
]

def build_rag_system(documents: list):
    """TODO: Build FAISS index with sentence-transformers and ground LLM query responses"""
    raise NotImplementedError("Implement RAG system logic here.")

def run():
    print("=" * 60)
    print(" Practical: 14. RAG + Embeddings + FAISS")
    print("=" * 60)
    print("Objective: Retrieval-Augmented Generation using all-MiniLM-L6-v2 embeddings, FAISS vector store, and LLaMA 3.2.")
    print("-" * 60)
    # TODO: Call your implementation function here
    print("Ready for implementation!")
    print("=" * 60)

if __name__ == "__main__":
    run()
