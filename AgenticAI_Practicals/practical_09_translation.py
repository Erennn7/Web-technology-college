from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(
    model = "llama3.2",
    temperature = 0.3
)

prompt =PromptTemplate.from_template(
    """ 
    Translate the following text into {language} 
    Text : {text}

    """
)

chain = prompt | llm 

text = input("Enter text : ")
language = input("Enter language. : ")

result = chain.invoke({
    "text":text,
    "language" : language
})

print(result)