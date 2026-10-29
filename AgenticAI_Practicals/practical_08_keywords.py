from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate 

llm = OllamaLLM(
    model = "llama3.2",
    temperature =0
)

prompt = PromptTemplate.from_template(
    """
    Extract only 5 important keywords from this sentence REturn only the keywords
    Sentence : {sentence}

    """
)

chain = prompt |llm 
sentence = input("Enter the sentence")

result = chain.invoke({
    "sentence" : sentence
})

print(result)