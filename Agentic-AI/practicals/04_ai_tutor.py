"""
Practical: 4. AI Tutor
Objective: Explain complex technical concepts using analogies, step-by-step breakdown, and an interactive quiz question.
"""
import sys
import os

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

TOPIC = "Gradient Descent and Backpropagation"

def teach_concept(topic: str) -> str:
    """TODO: Implement Feynman-style interactive tutor"""
    raise NotImplementedError("Implement AI Tutor logic here.")

def run():
    print("=" * 60)
    print(" Practical: 4. AI Tutor")
    print("=" * 60)
    print("Objective: Explain complex technical concepts using analogies, step-by-step breakdown, and an interactive quiz question.")
    print("-" * 60)
    # TODO: Call your implementation function here
    print("Ready for implementation!")
    print("=" * 60)

if __name__ == "__main__":
    run()
