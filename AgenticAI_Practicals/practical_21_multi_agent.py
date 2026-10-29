from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="llama3.2",
    temperature=0.2
)

task =input("Enter task: ")


expert= llm.invoke(
    f"""
    Assign expert for this task :{task}
    Options:
    1.MATH
    2.PYTHON
    3.WRITING
    
    Return only specialist name 
    """
).strip()

print(f"Specialist assigned : {expert}")

answer = llm.invoke(
    f"""
    You are an {expert} expert
    Solve this {task}
    """
)

print("Answer : ")
print(answer)