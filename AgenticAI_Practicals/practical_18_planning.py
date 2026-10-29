from langchain_ollama import OllamaLLM


llm =OllamaLLM(
    model ="llama3.2",
    temperature=0.1
)

task = input("Enter the task")


plan =llm.invoke(
    f"""
    Generate steps to complete this task : {task}
    """
)

print("plan : ")
print(plan)


execution = llm.invoke(
    f"""
    Execute this conceptually based on the plan : {plan}
    """
)

print("Execution ")
print(execution)
