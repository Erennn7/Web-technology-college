from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(
    model = "llama3.2",
    temperature = 0.3
)

prompt = PromptTemplate.from_template(
    """
    Classify the sentiment of the following text 
    Return only one of :
    Positve / Negative / Neutral

    Text : {text}
    """
)

chain = prompt | llm 


text = input("Enter text: ")

result = chain.invoke({
    "text" : text
})

print(" Sentiment analysis :  ")
print(result)