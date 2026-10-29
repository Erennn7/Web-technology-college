"""
Practical: 19. ReAct + Tool
Objective: ReAct (Reasoning + Acting) loop: Thought -> Action -> Observation -> Final Answer.
"""
import sys
import os

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

USER_QUESTION = "Calculate the total area of two squares with sides 12 and 15, then multiply by 2."

def run_react_agent(query: str):
    """TODO: Implement ReAct reasoning trace loop with tools"""
    raise NotImplementedError("Implement ReAct agent logic here.")

def run():
    print("=" * 60)
    print(" Practical: 19. ReAct + Tool")
    print("=" * 60)
    print("Objective: ReAct (Reasoning + Acting) loop: Thought -> Action -> Observation -> Final Answer.")
    print("-" * 60)
    # TODO: Call your implementation function here
    print("Ready for implementation!")
    print("=" * 60)

if __name__ == "__main__":
    run()
