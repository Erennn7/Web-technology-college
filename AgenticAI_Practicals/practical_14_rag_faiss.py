from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


data = [
    "College starts at 9am ",
    "Library closes at 10pm ",
    "Canteen has great food",
    "Exam hall on 2nd floor"
]

embeddings =HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_texts(
    data,
    embeddings
)
question=input("Ask a question")

docs = db.similarity_search(
    question,
    k=1
)

print(docs[0].page_content)