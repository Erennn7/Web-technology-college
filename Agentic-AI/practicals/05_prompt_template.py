"""
Practical: 5. Prompt Template
Objective: Demonstrate LangChain PromptTemplate and ChatPromptTemplate with dynamic variable interpolation.
"""
import sys
import os

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

CODE_SNIPPET = "for i in range(n): for j in range(n): print(i, j)"

def demonstrate_prompt_template():
    """TODO: Construct PromptTemplate / ChatPromptTemplate and execute with LLaMA 3.2"""
    raise NotImplementedError("Implement prompt template logic here.")

def run():
    print("=" * 60)
    print(" Practical: 5. Prompt Template")
    print("=" * 60)
    print("Objective: Demonstrate LangChain PromptTemplate and ChatPromptTemplate with dynamic variable interpolation.")
    print("-" * 60)
    # TODO: Call your implementation function here
    print("Ready for implementation!")
    print("=" * 60)

if __name__ == "__main__":
    run()
