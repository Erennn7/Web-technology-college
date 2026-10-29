"""
Practical: 11. AI Travel Agent
Objective: Autonomous travel agent that generates a day-by-day itinerary with budget estimation.
"""
import sys
import os

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

DESTINATION = "Tokyo, Japan"
DAYS = 5
BUDGET = "$2500"

def plan_travel(destination: str, days: int, budget: str) -> str:
    """TODO: Implement travel planning agent"""
    raise NotImplementedError("Implement travel agent logic here.")

def run():
    print("=" * 60)
    print(" Practical: 11. AI Travel Agent")
    print("=" * 60)
    print("Objective: Autonomous travel agent that generates a day-by-day itinerary with budget estimation.")
    print("-" * 60)
    # TODO: Call your implementation function here
    print("Ready for implementation!")
    print("=" * 60)

if __name__ == "__main__":
    run()
