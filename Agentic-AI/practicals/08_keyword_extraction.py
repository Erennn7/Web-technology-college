"""
Practical: 8. Keyword Extraction
Objective: Extract key entities, technical terms, and topics from unformatted text, returning structured JSON tags.
"""
import sys
import os

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

TEXT = "OpenAI, Meta, and Google DeepMind are competing in frontier generative models, RAG architectures, and autonomous agent systems."

def extract_keywords(text: str) -> list:
    """TODO: Extract structured keywords and entities"""
    raise NotImplementedError("Implement keyword extraction logic here.")

def run():
    print("=" * 60)
    print(" Practical: 8. Keyword Extraction")
    print("=" * 60)
    print("Objective: Extract key entities, technical terms, and topics from unformatted text, returning structured JSON tags.")
    print("-" * 60)
    # TODO: Call your implementation function here
    print("Ready for implementation!")
    print("=" * 60)

if __name__ == "__main__":
    run()
