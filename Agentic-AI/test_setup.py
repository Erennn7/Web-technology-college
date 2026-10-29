"""
Agentic AI Practicals - Environment Verification Script
Runs end-to-end checks on all required components:
1. Python Version & Virtual Environment
2. Core Libraries (LangChain, FAISS, Sentence-Transformers, Transformers, Torch)
3. Ollama Server Status & llama3.2 Model Inference
4. Sentence-Transformers (all-MiniLM-L6-v2) & FAISS Vector Indexing
5. External APIs (wttr.in Weather API)
"""
import sys
import os

def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def test_environment():
    print_header("1. Checking Python & Environment")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Python Executable: {sys.executable}")
    is_venv = sys.prefix != sys.base_prefix
    print(f"Running inside Virtual Environment: {'YES' if is_venv else 'NO'}")
    assert sys.version_info >= (3, 10), "Python 3.10+ required"
    print("[PASS] Python environment is ready.")

def test_imports():
    print_header("2. Checking Required Packages")
    packages = [
        ("langchain", "LangChain"),
        ("langchain_core", "LangChain Core"),
        ("langchain_ollama", "LangChain Ollama"),
        ("langchain_community", "LangChain Community"),
        ("langchain_text_splitters", "LangChain Text Splitters"),
        ("langchain_huggingface", "LangChain HuggingFace"),
        ("ollama", "Ollama Python Client"),
        ("sentence_transformers", "Sentence Transformers"),
        ("faiss", "FAISS Vector DB"),
        ("transformers", "Hugging Face Transformers"),
        ("torch", "PyTorch"),
        ("PIL", "Pillow"),
        ("requests", "Requests (Weather API)")
    ]
    
    all_ok = True
    for mod_name, display_name in packages:
        try:
            __import__(mod_name)
            print(f" [PASS] {display_name:<30} (imported successfully)")
        except ImportError as e:
            print(f" [FAIL] {display_name:<30} - {e}")
            all_ok = False
            
    if not all_ok:
        print("\nSome packages are not yet installed or still downloading.")
    return all_ok

def test_ollama():
    print_header("3. Checking Ollama & llama3.2 Model")
    import ollama
    try:
        models_resp = ollama.list()
        # Handle dict or list structure
        models = [m.model if hasattr(m, 'model') else m.get('name', '') for m in getattr(models_resp, 'models', [])]
        print(f"Available Ollama models: {models}")
        
        has_llama = any("llama3.2" in m for m in models)
        if not has_llama:
            print(" [WARNING] llama3.2 is not listed yet (it might still be pulling).")
            return False
            
        print(" [PASS] llama3.2 model found. Testing prompt: 'What is artificial intelligence?'")
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": "What is artificial intelligence? Answer in 1 short sentence."}],
            options={"temperature": 0.2}
        )
        answer = response["message"]["content"]
        print(f" [Ollama Response]: {answer.strip()}")
        print(" [PASS] Ollama inference is working smoothly!")
        return True
    except Exception as e:
        print(f" [FAIL] Could not connect to Ollama: {e}")
        return False

def test_embeddings_and_faiss():
    print_header("4. Checking Embeddings (all-MiniLM-L6-v2) & FAISS")
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np

        print(" Loading 'all-MiniLM-L6-v2'...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [
            "Agentic AI uses autonomous planning and tool use.",
            "Retrieval Augmented Generation grounds LLMs with vector search.",
            "Weather in London is rainy and cold today."
        ]
        embeddings = embedder.encode(texts)
        embeddings = np.array(embeddings).astype("float32")
        
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        print(f" [PASS] FAISS index created with dimension {dim} and {index.ntotal} vectors.")
        
        # Test query
        query_text = ["RAG and vector databases"]
        q_emb = np.array(embedder.encode(query_text)).astype("float32")
        distances, indices = index.search(q_emb, k=1)
        matched_text = texts[indices[0][0]]
        print(f" [PASS] Search Test: Query '{query_text[0]}' -> Top Match: '{matched_text}'")
        return True
    except Exception as e:
        print(f" [FAIL] Embeddings/FAISS check error: {e}")
        return False

def test_weather_api():
    print_header("5. Checking Weather API (wttr.in)")
    import requests
    try:
        res = requests.get("https://wttr.in/London?format=j1", timeout=5)
        if res.status_code == 200:
            data = res.json()
            temp = data["current_condition"][0]["temp_C"]
            desc = data["current_condition"][0]["weatherDesc"][0]["value"]
            print(f" [PASS] wttr.in responded. London: {temp}°C, {desc}")
            return True
        else:
            print(f" [FAIL] wttr.in returned status code {res.status_code}")
            return False
    except Exception as e:
        print(f" [FAIL] wttr.in connection error: {e}")
        return False

def main():
    print("\n" + "#" * 60)
    print("      AGENTIC AI LAB ENVIRONMENT VERIFICATION")
    print("#" * 60)
    
    test_environment()
    pkgs_ok = test_imports()
    ollama_ok = test_ollama()
    embed_ok = test_embeddings_and_faiss()
    weather_ok = test_weather_api()
    
    print("\n" + "=" * 60)
    print(" VERIFICATION SUMMARY")
    print("=" * 60)
    print(f" 1. Packages & Dependencies : {'READY' if pkgs_ok else 'INSTALLING / PENDING'}")
    print(f" 2. Ollama & llama3.2       : {'READY' if ollama_ok else 'PULLING / PENDING'}")
    print(f" 3. Embeddings & FAISS      : {'READY' if embed_ok else 'PENDING'}")
    print(f" 4. Weather API (wttr.in)   : {'READY' if weather_ok else 'PENDING'}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
