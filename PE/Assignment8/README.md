# Walchand Institute of Technology, Solapur
## Department of Information Technology
### Course: Program Elective – V (Prompt Engineering)

---

# Assignment No 8: Integrated Coding Assignment
## AI-Powered Content Creation and Analysis System

---

## 🎯 Objective
Develop a Python-based AI application using an LLM/API to demonstrate prompt engineering, content generation, podcast planning, text analysis, and parameter experimentation.

---

## 📁 Repository & File Structure

```
PE/Assignment8/
├── app.py              # Interactive Streamlit Web Application
├── cli_main.py         # Command-Line Interface (CLI) runner
├── prompts.py          # Structured prompt engineering design schemas
├── llm_handler.py      # Unified LLM provider wrapper (Gemini, OpenAI, Mock Mode)
└── README.md           # Documentation & Assignment Report
```

---

## 🛠️ Key Features & Implementation Details

### 1. Prompt Design Architecture (`prompts.py`)
All prompts in this system are constructed using structured design patterns:
- **Role:** Assigns a specific expert persona (e.g., *Creative Storyteller*, *Executive Podcast Producer*, *NLP Linguist*).
- **Context:** Defines domain scope, target audience, and intent.
- **Constraints:** Enforces line counts, formatting rules, tone, and strict output boundaries.
- **Output Format:** Specifies exact schemas (Markdown headings, structured JSON, bulleted lists).

### 2. AI Content Generation (`app.py` & `cli_main.py`)
Generates high-quality AI content across multiple creative media formats:
- **AI Story:** 3-paragraph narrative arc (Hook, Conflict, Resolution) with a clear takeaway.
- **AI Poem:** 4-stanza evocative poem with structured rhyme schemes (AABB/ABAB).
- **Social Media Posts:** Platform-tailored posts for Twitter/X (under 280 chars), LinkedIn (professional insight with CTA), and Instagram (caption with emojis & hashtags).

### 3. Podcast Planning
Generates a complete episode production plan:
- **Episode Title:** Catchy and click-worthy title.
- **Show Notes / Description:** 100–150 word summary for Spotify / Apple Podcasts.
- **Ideal Guest Profile:** Target role, required expertise, and rationale.
- **5 Core Interview Questions:** Ranging from introductory context to deep dive and future outlook.

### 4. Text Analysis & NLP
Performs multi-faceted text analysis on user-supplied text:
- **Sentiment Analysis:** Calculates sentiment score from `-1.0` (extremely negative) to `+1.0` (extremely positive), assigns label (`Positive`, `Negative`, `Neutral`), and provides rationale.
- **Keyword Extraction:** Identifies 5–10 domain keyphrases along with confidence relevance scores.
- **Visual Analytics:** Interactive bar chart displaying keyword relevance distribution.

### 5. LLM Parameter Experimentation
Compares LLM outputs across different decoding parameters:
- **Temperature ($T$):** Controls randomness/creativity ($0.0 \le T \le 1.5$).
  - Low ($0.1$): Deterministic, repeatable, concise.
  - Balanced ($0.7$): Optimal blend of creativity and structure.
  - High ($1.2$): High creative variation and unexpected phrasing.
- **Top-P (Nucleus Sampling):** Controls cumulative probability threshold ($0.1 \le P \le 1.0$).

---

## 🚀 How to Run the Application

### Prerequisites
Install the required packages using pip:
```bash
pip install streamlit google-genai openai matplotlib pandas python-dotenv
```

### Option A: Launch Interactive Web Application (Streamlit)
```bash
cd PE/Assignment8
streamlit run app.py
```
*The app will automatically open in your browser at `http://localhost:8501`.*

### Option B: Run via Command Line Interface (CLI)
```bash
cd PE/Assignment8
python3 cli_main.py --topic "Artificial Intelligence in Healthcare"
```

To run with a custom API key:
```bash
python3 cli_main.py --topic "Quantum Computing" --provider gemini --api-key "YOUR_GEMINI_API_KEY"
```

*Note: If no API key is provided, the system automatically runs in **Mock Engine Mode**, allowing full demonstration without external API dependencies.*

---

## 📊 Summary of Findings & LLM Parameter Observations

| Parameter Setting | Temperature ($T$) | Top-P ($P$) | Observed Behavior & Best Use Cases |
|---|---|---|---|
| **Deterministic** | `0.1` | `0.5` | Highly predictable, strict adherence to rules. Best for **JSON extraction**, **sentiment analysis**, and **coding**. |
| **Balanced** | `0.7` | `0.9` | Coherent, engaging, and natural tone. Best for **podcast planning**, **story generation**, and **article writing**. |
| **High Creative** | `1.2` | `0.98` | Diverse vocabulary, figurative metaphors. Best for **brainstorming**, **poetry**, and **artistic content**. |
