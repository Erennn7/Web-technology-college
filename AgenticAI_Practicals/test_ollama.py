from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="llama3.2",
    temperature=0.7
)

response = llm.invoke("Explain Artificial Intelligence in simple words.")
print(response)
