from langchain_core.tools import tool


@tool 
def multiply(a: int, b: int):
    """Multiply two numbers"""
    return a*b 

a=int(input("Enter first number"))
b=int(input("Enter second number"))

print("Thought need multiplication")
print("ACtion: multiply")

result = multiply.invoke({
    "a":a,
    "b":b
})

print("Final Result : ")
print(result)