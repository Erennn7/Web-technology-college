"""
Practical: 3. Interview Question Generator
Objective: Generate role-specific technical and behavioral interview questions with evaluation criteria.
"""
import sys
import os

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

ROLE = "Senior Full Stack AI Developer"
TECH_STACK = "Python, LangChain, React, Vector DBs"

def generate_interview_questions(role: str, tech_stack: str) -> str:
    """TODO: Generate customized interview questions"""
    raise NotImplementedError("Implement question generator logic here.")

def run():
    print("=" * 60)
    print(" Practical: 3. Interview Question Generator")
    print("=" * 60)
    print("Objective: Generate role-specific technical and behavioral interview questions with evaluation criteria.")
    print("-" * 60)
    # TODO: Call your implementation function here
    print("Ready for implementation!")
    print("=" * 60)

if __name__ == "__main__":
    run()
