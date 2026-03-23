# Simple Reflex Agent for Vacuum Cleaner

def simple_reflex_agent(percept):
    location, status = percept

    if status == "Dirty":
        return "clean"
    elif location == "A":
        return "Move Right"
    elif location == "B":
        return "Move Left"

# Example percepts
percepts = [
    ("A", "Dirty"),
    ("A", "Clean"),
    ("B", "Dirty"),
    ("B", "Clean")
]

# Run agent
for percept in percepts:
    action = simple_reflex_agent(percept)
    print(f"Percept: {percept} -> Action: {action}")