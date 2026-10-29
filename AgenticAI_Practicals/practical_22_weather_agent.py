import requests
from langchain_ollama import OllamaLLM


llm = OllamaLLM(
    model ="llama3.2",
    temperature=0.2
)

city =input("Enter city : ")

url =f"https://wttr.in/{city}?format=j1"

response = requests.get(url)

data =response.json()

temp =data["current_condition"][0]["temp_C"]

weather = data["current_condition"][0]["weatherDesc"][0]["value"]

prompt = f"""
You are a weather assistant and give some advice to people in {city}

Current weather : {weather}
Temperature : {temp}
"""

print(llm.invoke(prompt))