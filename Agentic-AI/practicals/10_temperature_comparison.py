"""
Practical: 10. Temperature Comparison
Objective: Demonstrate the effect of LLM temperature (0.0 vs 0.7 vs 1.2) on determinism vs creativity.
"""
import sys
import os

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

PROMPT = "Invent a creative name and tagline for an autonomous AI travel companion."

def compare_temperatures(prompt: str, temperatures=[0.0, 0.7, 1.2]):
    """TODO: Run the same prompt at different temperatures and compare outputs"""
    raise NotImplementedError("Implement temperature comparison logic here.")

def run():
    print("=" * 60)
    print(" Practical: 10. Temperature Comparison")
    print("=" * 60)
    print("Objective: Demonstrate the effect of LLM temperature (0.0 vs 0.7 vs 1.2) on determinism vs creativity.")
    print("-" * 60)
    # TODO: Call your implementation function here
    print("Ready for implementation!")
    print("=" * 60)

if __name__ == "__main__":
    run()
