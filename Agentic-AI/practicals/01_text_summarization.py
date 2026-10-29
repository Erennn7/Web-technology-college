"""
Practical: 1. Text Summarization
Objective: Summarize long-form text into concise bullet points and an executive summary using Ollama (llama3.2).
"""
import sys
import os

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SAMPLE_TEXT = """
Artificial Intelligence and Agentic workflows are transforming software engineering.
Modern agents can perceive their environment, reason over complex goals, break down challenges into sub-tasks, invoke external tools, and iteratively self-correct.
"""

def summarize_text(text: str) -> str:
    """TODO: Implement text summarization using LangChain or Ollama API"""
    raise NotImplementedError("Implement your summarization logic here.")

def run():
    print("=" * 60)
    print(" Practical: 1. Text Summarization")
    print("=" * 60)
    print("Objective: Summarize long-form text into concise bullet points and an executive summary using Ollama (llama3.2).")
    print("-" * 60)
    # TODO: Call your implementation function here
    print("Ready for implementation!")
    print("=" * 60)

if __name__ == "__main__":
    run()
