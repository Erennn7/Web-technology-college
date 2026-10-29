from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool


@tool
def refund():
    """Handle refund request """ 
    return "Refund intitated"

@tool 
def technical():
    """Handle technical issue """
    return "Technical ticket raise"


llm = OllamaLLM(
    model ="llama3.2",
    temperature=0.1
)

query = input("Customer query").lower()


if "refund" in query:
    result  = refund.invoke({})

elif "error" in query or "not working" in query:
    result = technical.invoke({})

else:
    result = llm.invoke(query)

print(result)
