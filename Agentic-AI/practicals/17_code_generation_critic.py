"""
Practical: 17. Code Generation + Critic
Objective: Two-phase agent: Coder writes solution -> Critic reviews edge cases & bugs -> Coder refines.
"""
import sys
import os

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

PROBLEM = "Write a Python function to find the longest palindromic substring in O(N^2) time."

def generate_and_critique(problem: str):
    """TODO: Implement actor-critic loop for code generation and refinement"""
    raise NotImplementedError("Implement code generator + critic logic here.")

def run():
    print("=" * 60)
    print(" Practical: 17. Code Generation + Critic")
    print("=" * 60)
    print("Objective: Two-phase agent: Coder writes solution -> Critic reviews edge cases & bugs -> Coder refines.")
    print("-" * 60)
    # TODO: Call your implementation function here
    print("Ready for implementation!")
    print("=" * 60)

if __name__ == "__main__":
    run()
