from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


llm = OllamaLLM(
    model = "llama3.2",
    temperature=0.5
)

prompt= PromptTemplate.from_template(
    """
    You are an AI Tutor 
    Explain {topic} to a {level} student

    Use :
    1.simple definition
    2.Example
    3.Important points
    """
)

chain = prompt | llm 

topic = input("Enter topic:")
level =input("enter level : ")

result = chain.invoke({
    "topic":topic,
    "level": level 
})

print(result)