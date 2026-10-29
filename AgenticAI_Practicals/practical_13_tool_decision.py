from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

@tool 
def difficulty(experience : int):
    """Select interview difficulty """
    if experience < 2:
        return "Easy"
    else :
        return "Advanced"


llm =OllamaLLM(
    model = "llama3.2",
    temperature=0.2
)

prompt = PromptTemplate.from_template(

    """
    Generate 3 interview questions
    Difficulty :{level}
    """
)

chain = prompt |llm 
years = int(input("Enter years of experience : "))

level= difficulty.invoke({
        "experience":years
    })

result = chain.invoke({
    "years" :years,
    "level" : level
})
print(result)