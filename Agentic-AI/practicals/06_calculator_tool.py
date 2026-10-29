"""
Practical: 6. Calculator Tool
Objective: Define a mathematical calculator tool for an agent to evaluate arithmetic and algebraic expressions safely.
"""
import sys
import os

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

EXPRESSIONS = ["25 * 4 + 100 / 5", "2 ** 8 + 45"]

def calculate(expression: str) -> float:
    """TODO: Implement safe calculator tool for agent tool-use"""
    raise NotImplementedError("Implement calculator logic here.")

def run():
    print("=" * 60)
    print(" Practical: 6. Calculator Tool")
    print("=" * 60)
    print("Objective: Define a mathematical calculator tool for an agent to evaluate arithmetic and algebraic expressions safely.")
    print("-" * 60)
    # TODO: Call your implementation function here
    print("Ready for implementation!")
    print("=" * 60)

if __name__ == "__main__":
    run()
