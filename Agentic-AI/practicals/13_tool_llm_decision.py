"""
Practical: 13. Tool + LLM Decision Making
Objective: LLM decides whether an external tool (e.g. calculator, current time, weather) is needed or direct answer.
"""
import sys
import os

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

QUERIES = ["What is the capital of France?", "Calculate 9876 * 5432", "What is the weather in Tokyo?"]

def decide_and_execute(query: str):
    """TODO: Implement tool selection and dispatch based on LLM decision"""
    raise NotImplementedError("Implement tool decision logic here.")

def run():
    print("=" * 60)
    print(" Practical: 13. Tool + LLM Decision Making")
    print("=" * 60)
    print("Objective: LLM decides whether an external tool (e.g. calculator, current time, weather) is needed or direct answer.")
    print("-" * 60)
    # TODO: Call your implementation function here
    print("Ready for implementation!")
    print("=" * 60)

if __name__ == "__main__":
    run()
