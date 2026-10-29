"""
Practical: 7. Document Chunking
Objective: Demonstrate text splitting strategies (RecursiveCharacterTextSplitter) with chunk size & overlap.
"""
import sys
import os

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SAMPLE_DOC = """LangChain provides text splitters to divide large documents into semantically coherent chunks for RAG vector indexing. Different strategies like character splitting, token splitting, and recursive splitting optimize chunk boundary retention."""

def chunk_document(document: str, chunk_size: int = 100, chunk_overlap: int = 20):
    """TODO: Implement document chunking using LangChain text splitters"""
    raise NotImplementedError("Implement document chunking logic here.")

def run():
    print("=" * 60)
    print(" Practical: 7. Document Chunking")
    print("=" * 60)
    print("Objective: Demonstrate text splitting strategies (RecursiveCharacterTextSplitter) with chunk size & overlap.")
    print("-" * 60)
    # TODO: Call your implementation function here
    print("Ready for implementation!")
    print("=" * 60)

if __name__ == "__main__":
    run()
