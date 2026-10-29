"""
Practical: 15. Classification + Tool + LLM
Objective: Classify incoming support tickets, trigger specific tool, and draft resolution.
"""
import sys
import os

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

TICKETS = ["I was charged twice for subscription #9942", "The API returns 500 error on POST /chat"]

def process_ticket(ticket: str):
    """TODO: Classify ticket -> call corresponding tool -> draft response"""
    raise NotImplementedError("Implement classification + tool logic here.")

def run():
    print("=" * 60)
    print(" Practical: 15. Classification + Tool + LLM")
    print("=" * 60)
    print("Objective: Classify incoming support tickets, trigger specific tool, and draft resolution.")
    print("-" * 60)
    # TODO: Call your implementation function here
    print("Ready for implementation!")
    print("=" * 60)

if __name__ == "__main__":
    run()
