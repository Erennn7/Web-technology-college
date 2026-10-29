from langchain_ollama import OllamaLLM

symptom_probability = {
    "mild":0.2,
    "moderate" : 0.5,
    "severe":0.7,
}

severity = input("Severity(mild/moderate/severe) : ").lower()

risk =symptom_probability[severity]

llm = OllamaLLM(
    model = "llama3.2",
    temperature = 0.1
)

if risk > 0.7:
    action = "Seek professional help "
else:
    action = "monitor symptoms"


result =llm.invoke(
    f"""
    {action}
    """
)
print(risk)
print(result)