from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

weather = input("Enter weather : ")
budget = input("Enter budget : ")

if weather.lower() == "rain":
    activity = "indoor activities"
elif budget <1000:
    activity = "local sigtseeing"
else:
    activity = "Premium outdoor activity"


prompt = PromptTemplate.from_template(
    """ You are an AI travel agent suggest me the activities for the foloowing 
    Activity type : {activity}
    Budget :{budget}
    Weather :{weather}
    Give 1.Activity 2. Cost 3.Short reason
    """

)

llm = OllamaLLM(
    model ="llama3.2"
    
)

chain = prompt | llm 

result = chain.invoke({
    "activity" : activity,
    "budget" :budget,
    "weather" : weather
})

print(result)