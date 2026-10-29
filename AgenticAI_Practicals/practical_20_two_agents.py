from langchain_ollama import OllamaLLM


llm = OllamaLLM(
    model="llama3.2",
    temperature=0.1
)

topic =input("Enter a topic: ")

research = llm.invoke(
    f"""
    Do deep research on {topic}
    """
)

print("Research : ")
print(research)

report = llm.invoke(
    f"""
    Convert this {research} into a presentable report 
    """
)

print("Report : ")
print(report)