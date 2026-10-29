from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


llm = OllamaLLM(
    model = "llama3.2",
    temperature =0.5
)

prompt = PromptTemplate.from_template(
    """
    Explain the following {topic} to a student """

)

chain = prompt | llm 

topic = input("Enter topic")

result = chain.invoke({
    "topic":  topic 
})

print(result)