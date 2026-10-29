"""
Agentic AI Practicals - Interactive Runner
Allows you to select and execute any practical from 1 to 23.
"""
import sys
import os
import importlib

PRACTICALS = [
    ("01_text_summarization", "Text Summarization"),
    ("02_sentiment_classification", "Sentiment Classification"),
    ("03_interview_question_generator", "Interview Question Generator"),
    ("04_ai_tutor", "AI Tutor"),
    ("05_prompt_template", "Prompt Template"),
    ("06_calculator_tool", "Calculator Tool"),
    ("07_document_chunking", "Document Chunking"),
    ("08_keyword_extraction", "Keyword Extraction"),
    ("09_translation", "Translation"),
    ("10_temperature_comparison", "Temperature Comparison"),
    ("11_ai_travel_agent", "AI Travel Agent"),
    ("12_study_planning_agent", "Study Planning Agent"),
    ("13_tool_llm_decision", "Tool + LLM Decision Making"),
    ("14_rag_embeddings_faiss", "RAG + Embeddings + FAISS"),
    ("15_classification_tool_llm", "Classification + Tool + LLM"),
    ("16_probability_decision_llm", "Probability + Decision Making + LLM"),
    ("17_code_generation_critic", "Code Generation + Critic"),
    ("18_planning_execution", "Planning + Execution"),
    ("19_react_tool", "ReAct + Tool"),
    ("20_two_agent_collaboration", "Two-Agent Collaboration"),
    ("21_multi_agent_routing", "Multi-Agent Routing"),
    ("22_weather_api_llm", "Weather API + LLM"),
    ("23_image_processing_agent", "Image Processing + Agent Decision"),
]

def print_menu():
    print("\n" + "=" * 65)
    print("           AGENTIC AI PRACTICALS - LAB SUITE")
    print("=" * 65)
    for i, (_, title) in enumerate(PRACTICALS, 1):
        print(f" [{i:2d}] {title}")
    print(" [ 0] Run Environment Verification (test_setup.py)")
    print(" [ q] Quit")
    print("=" * 65)

def run_practical(choice):
    if choice == "0":
        import test_setup
        test_setup.main()
        return

    try:
        num = int(choice)
        if 1 <= num <= len(PRACTICALS):
            mod_name, title = PRACTICALS[num - 1]
            module_path = f"practicals.{mod_name}"
            mod = importlib.import_module(module_path)
            if hasattr(mod, "run"):
                mod.run()
            else:
                print(f"Error: {mod_name} does not define a run() function.")
        else:
            print("Invalid practical number. Choose 1-23.")
    except Exception as e:
        print(f"Execution error: {e}")

def main():
    while True:
        print_menu()
        choice = input("Enter practical number to run (or 'q' to quit): ").strip().lower()
        if choice in ("q", "quit", "exit"):
            print("Goodbye!")
            break
        run_practical(choice)
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
