"""
Practical: 22. Weather API + LLM
Objective: Query live wttr.in weather API for any city, and synthesize an agent advisory with LLaMA 3.2.
"""
import sys
import os

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

CITIES = ["Mumbai", "New York", "Tokyo"]

def get_weather_advisory(city: str):
    """TODO: Fetch wttr.in API data -> pass to LLaMA 3.2 to generate recommendations"""
    raise NotImplementedError("Implement weather agent logic here.")

def run():
    print("=" * 60)
    print(" Practical: 22. Weather API + LLM")
    print("=" * 60)
    print("Objective: Query live wttr.in weather API for any city, and synthesize an agent advisory with LLaMA 3.2.")
    print("-" * 60)
    # TODO: Call your implementation function here
    print("Ready for implementation!")
    print("=" * 60)

if __name__ == "__main__":
    run()
