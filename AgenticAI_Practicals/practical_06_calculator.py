from langchain_core.tools import tool 


@tool 
def add(a: int, b: int) ->int : 
    """Add two numbers """
    return a + b



a = input("Enter number 1: ")
b = input("Enter number 2: ")

result =add.invoke({
    "a":   a,
    "b": b
})

print(result)