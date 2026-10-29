"""
Practical: 2. Sentiment Classification
Objective: Classify user reviews into Positive, Negative, or Neutral with confidence and reasoning.
"""
import sys
import os

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

REVIEWS = [
    "The battery life is amazing and speed is unbelievable!",
    "Broken on arrival, horrible customer support.",
    "The item arrived on Tuesday as expected."
]

def classify_sentiment(review: str) -> dict:
    """TODO: Implement sentiment classification with structured output"""
    raise NotImplementedError("Implement your classification logic here.")

def run():
    print("=" * 60)
    print(" Practical: 2. Sentiment Classification")
    print("=" * 60)
    print("Objective: Classify user reviews into Positive, Negative, or Neutral with confidence and reasoning.")
    print("-" * 60)
    # TODO: Call your implementation function here
    print("Ready for implementation!")
    print("=" * 60)

if __name__ == "__main__":
    run()
