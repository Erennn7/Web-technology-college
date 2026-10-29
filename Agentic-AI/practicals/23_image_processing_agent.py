"""
Practical: 23. Image Processing + Agent Decision
Objective: Hugging Face Transformers image classification / vision pipeline -> agent decision making.
"""
import sys
import os

# Add parent directory to path for config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

IMAGE_PATH = "sample.jpg"

def process_image_and_decide(image_path: str):
    """TODO: Run vision pipeline -> LLM agent decides action based on detected objects/scene"""
    raise NotImplementedError("Implement vision agent logic here.")

def run():
    print("=" * 60)
    print(" Practical: 23. Image Processing + Agent Decision")
    print("=" * 60)
    print("Objective: Hugging Face Transformers image classification / vision pipeline -> agent decision making.")
    print("-" * 60)
    # TODO: Call your implementation function here
    print("Ready for implementation!")
    print("=" * 60)

if __name__ == "__main__":
    run()
