from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


llm = OllamaLLM(
    model = "llama3.2",
    temperature = 0.5
)

prompt = PromptTemplate.from_template(

    """
    Give me interview questions for the following topic  and number them 1 to 5 
    Topic : {topic}
    """
)

chain = prompt | llm 

topic = input("Enter topic:")

result = chain.invoke({
    "topic": topic
})

print(result)