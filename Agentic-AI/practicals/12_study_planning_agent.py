"""
Practical: 12. Study Planning Agent
Objective: Generate a prioritized, spaced-repetition study schedule given exam date and syllabus.
"""
import sys
import os

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SUBJECT = "Distributed Systems and Cloud Computing"
EXAM_DAYS_LEFT = 14

def create_study_plan(subject: str, days_left: int) -> str:
    """TODO: Implement study schedule generator"""
    raise NotImplementedError("Implement study plan logic here.")

def run():
    print("=" * 60)
    print(" Practical: 12. Study Planning Agent")
    print("=" * 60)
    print("Objective: Generate a prioritized, spaced-repetition study schedule given exam date and syllabus.")
    print("-" * 60)
    # TODO: Call your implementation function here
    print("Ready for implementation!")
    print("=" * 60)

if __name__ == "__main__":
    run()
