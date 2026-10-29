from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

low = OllamaLLM(
    model = "llama3.2",
    temperature = 0.2
)

high = OllamaLLM(
    model="llama3.2",
    temperature=0.9
)

prompt = "Write a creative sentence about AI "

print("Low : ")
print(low.invoke(prompt))

print("High : ")
print(high.invoke(prompt))