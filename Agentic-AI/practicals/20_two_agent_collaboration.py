"""
Practical: 20. Two-Agent Collaboration
Objective: Two specialized agents (e.g., Software Architect & Security Auditor) collaborate iteratively.
"""
import sys
import os

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

TOPIC = "Design a secure JWT authentication system with refresh token rotation."

def collaborate_agents(topic: str, rounds: int = 2):
    """TODO: Alternate dialogue and refinement between Agent A and Agent B"""
    raise NotImplementedError("Implement two-agent collaboration logic here.")

def run():
    print("=" * 60)
    print(" Practical: 20. Two-Agent Collaboration")
    print("=" * 60)
    print("Objective: Two specialized agents (e.g., Software Architect & Security Auditor) collaborate iteratively.")
    print("-" * 60)
    # TODO: Call your implementation function here
    print("Ready for implementation!")
    print("=" * 60)

if __name__ == "__main__":
    run()
