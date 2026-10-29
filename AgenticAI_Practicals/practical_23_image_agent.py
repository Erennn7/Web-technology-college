import os
from transformers import pipeline 

classifier = pipeline(
    "image-classification"
)

image = "image7.jpg"

result = classifier(image)

label = result[0]["label"]
score = result[0]["score"]

print(f"Identified : {label} with {score}")

if score > 0.7:
    print("Agent action : Accept prediction")
else:
    print("Agent action : Request human review")