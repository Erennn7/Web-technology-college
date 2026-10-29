"""
Practical: 18. Planning + Execution
Objective: Decompose a complex multi-step objective into subtasks and execute each step maintaining state.
"""
import sys
import os

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

OBJECTIVE = "Research modern vector databases, compare FAISS vs Chroma vs Pinecone, and produce a comparison table."

def plan_and_execute(objective: str):
    """TODO: Step 1: Generate plan steps. Step 2: Sequentially execute each sub-task."""
    raise NotImplementedError("Implement plan & execute agent logic here.")

def run():
    print("=" * 60)
    print(" Practical: 18. Planning + Execution")
    print("=" * 60)
    print("Objective: Decompose a complex multi-step objective into subtasks and execute each step maintaining state.")
    print("-" * 60)
    # TODO: Call your implementation function here
    print("Ready for implementation!")
    print("=" * 60)

if __name__ == "__main__":
    run()
