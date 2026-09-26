"""
LLM Interface Handler
Supports:
1. Google Gemini API (google-genai)
2. OpenAI API (openai)
3. Offline Simulation Mode
"""

import os
import json
import time
import re
from typing import Dict, Any, Tuple, Optional

HAS_GENAI = False
HAS_OPENAI = False

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class LLMHandler:
    def __init__(self, provider: str = "auto", api_key: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        
        if self.provider == "auto":
            if (os.environ.get("GEMINI_API_KEY") or (self.api_key and "AIza" in self.api_key)) and HAS_GENAI:
                self.provider = "gemini"
            elif (os.environ.get("OPENAI_API_KEY") or (self.api_key and "sk-" in self.api_key)) and HAS_OPENAI:
                self.provider = "openai"
            elif HAS_GENAI and self.api_key:
                self.provider = "gemini"
            elif HAS_OPENAI and self.api_key:
                self.provider = "openai"
            else:
                self.provider = "mock"

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        model_name: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()
        
        if self.provider == "mock" or not self.api_key:
            response_text = self._mock_generation(prompt, temperature, top_p)
            elapsed = round(time.time() - start_time, 3)
            meta = {
                "provider": "Offline Engine",
                "model": "rule-based-v1",
                "temperature": temperature,
                "top_p": top_p,
                "latency_seconds": elapsed,
                "status": "Success"
            }
            return response_text, meta

        if self.provider == "gemini":
            try:
                client = genai.Client(api_key=self.api_key)
                selected_model = model_name or "gemini-2.5-flash"
                config = types.GenerateContentConfig(
                    temperature=float(temperature),
                    top_p=float(top_p)
                )
                res = client.models.generate_content(
                    model=selected_model,
                    contents=prompt,
                    config=config
                )
                elapsed = round(time.time() - start_time, 3)
                meta = {
                    "provider": "Google Gemini",
                    "model": selected_model,
                    "temperature": temperature,
                    "top_p": top_p,
                    "latency_seconds": elapsed,
                    "status": "Success"
                }
                return res.text, meta
            except Exception as e:
                response_text = self._mock_generation(prompt, temperature, top_p)
                elapsed = round(time.time() - start_time, 3)
                meta = {
                    "provider": f"Offline Fallback ({type(e).__name__})",
                    "model": "mock-fallback",
                    "temperature": temperature,
                    "top_p": top_p,
                    "latency_seconds": elapsed,
                    "status": f"Fallback: {str(e)}"
                }
                return response_text, meta

        if self.provider == "openai":
            try:
                client = openai.OpenAI(api_key=self.api_key)
                selected_model = model_name or "gpt-4o-mini"
                res = client.chat.completions.create(
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=float(temperature),
                    top_p=float(top_p)
                )
                elapsed = round(time.time() - start_time, 3)
                content = res.choices[0].message.content or ""
                meta = {
                    "provider": "OpenAI",
                    "model": selected_model,
                    "temperature": temperature,
                    "top_p": top_p,
                    "latency_seconds": elapsed,
                    "status": "Success"
                }
                return content, meta
            except Exception as e:
                response_text = self._mock_generation(prompt, temperature, top_p)
                elapsed = round(time.time() - start_time, 3)
                meta = {
                    "provider": f"Offline Fallback ({type(e).__name__})",
                    "model": "mock-fallback",
                    "temperature": temperature,
                    "top_p": top_p,
                    "latency_seconds": elapsed,
                    "status": f"Fallback: {str(e)}"
                }
                return response_text, meta

        response_text = self._mock_generation(prompt, temperature, top_p)
        elapsed = round(time.time() - start_time, 3)
        return response_text, {
            "provider": "Offline Engine",
            "model": "mock",
            "temperature": temperature,
            "top_p": top_p,
            "latency_seconds": elapsed,
            "status": "Success"
        }

    def _mock_generation(self, prompt: str, temperature: float, top_p: float) -> str:
        lower_prompt = prompt.lower()
        topic_match = re.search(r"User Input / Topic:\s*(.*)", prompt, re.IGNORECASE)
        topic = topic_match.group(1).strip() if topic_match else "Artificial Intelligence in Education"

        if "story" in lower_prompt:
            return (
                f"Title: The Dawn of {topic.title()}\n\n"
                f"Story:\n"
                f"The sun rose over the horizon as Maya stood gazing at the burgeoning breakthrough of {topic}. "
                f"For years, researchers had questioned whether such innovation could truly transform society. "
                f"Yet today, as the first prototype activated, the air vibrated with unmatched energy and quiet hope.\n\n"
                f"Challenges quickly emerged when unexpected complexities strained the system. Maya worked tirelessly through "
                f"the night, tweaking algorithms and refining prompts to ensure stability. Her determination proved pivotal when "
                f"the final test delivered flawless performance.\n\n"
                f"Looking back, Maya realized that human creativity paired with powerful technology was not just a tool, "
                f"but a catalyst for endless human potential.\n\n"
                f"Moral / Takeaway: True innovation is born when human perseverance guides technological discovery."
            )
        
        elif "poem" in lower_prompt:
            return (
                f"Title: Whispers of {topic.title()}\n\n"
                f"A spark of light in silent space,\n"
                f"Unfolding truth with gentle grace.\n"
                f"In thoughts of {topic.lower()} we find,\n"
                f"The wonders of a curious mind.\n\n"
                f"Through stormy seas and quiet skies,\n"
                f"A brand new era starts to rise.\n"
                f"With eager hands and hearts awake,\n"
                f"Great promises we strive to make.\n\n"
                f"Though paths may wind and shadow fall,\n"
                f"Clear wisdom echoes through it all.\n"
                f"And as the stars illuminate the night,\n"
                f"Our vision shines in steady light."
            )
            
        elif "social media" in lower_prompt or "twitter" in lower_prompt:
            return (
                f"--- TWITTER / X POST ---\n"
                f"Exploring the future of #{topic.replace(' ', '')}. The convergence of AI and domain expertise is redefining modern workflows. What is your perspective?\n"
                f"Hashtags: #{topic.replace(' ', '')} #Innovation #FutureTech #AI\n\n"
                f"--- LINKEDIN POST ---\n"
                f"Key Industry Shifts in {topic.title()}\n\n"
                f"As technology accelerates, professionals and decision makers are discovering that {topic} represents a fundamental paradigm shift.\n\n"
                f"Key Takeaways:\n"
                f"1. Increased operational efficiency & throughput\n"
                f"2. Empowering teams with intelligent automation\n"
                f"3. Strategic advantage for forward-looking organizations\n\n"
                f"How is your organization adapting to this evolution? Share your insights below.\n"
                f"Hashtags: #{topic.replace(' ', '')} #TechLeadership #PromptEngineering #AI\n\n"
                f"--- INSTAGRAM CAPTION ---\n"
                f"Unlocking potential with {topic.title()}. From concept validation to real-world deployment, the journey of technological iteration continues.\n\n"
                f"Hashtags: #{topic.replace(' ', '')} #TechUpdate #Innovation #Technology"
            )

        elif "podcast" in lower_prompt or "interview" in lower_prompt:
            return (
                f"Episode Title: Unlocking {topic.title()}: Strategies, Breakthroughs, and Industry Trends\n\n"
                f"Description:\n"
                f"In this episode, we examine the practical applications of {topic} with leading industry experts. "
                f"We explore fundamental principles, practical case studies, and structured prompt engineering methodologies "
                f"shaping modern applications. Designed for developers, researchers, and strategists seeking actionable insights.\n\n"
                f"Target Guest Profile:\n"
                f"- Ideal Role: Senior AI Engineering Lead\n"
                f"- Expertise: Prompt Engineering, LLM Integration, System Architecture\n"
                f"- Rationale: Demonstrated track record in deploying scalable AI systems.\n\n"
                f"Interview Questions:\n"
                f"1. Could you provide a brief overview of your background and what drew you to {topic}?\n"
                f"2. What foundational concepts are essential when approaching {topic} today?\n"
                f"3. What was a major technical bottleneck or challenge encountered in your recent deployments?\n"
                f"4. How do hyperparameter adjustments and prompt structure impact system reliability?\n"
                f"5. What major advancements do you anticipate in this domain over the next 3 to 5 years?"
            )

        elif "sentiment" in lower_prompt or "analysis" in lower_prompt or "json" in lower_prompt:
            return json.dumps({
                "sentiment": {
                    "score": 0.82,
                    "label": "Positive",
                    "explanation": f"The input text demonstrates optimism and constructive perspective regarding {topic}."
                },
                "keywords": [
                    {"keyword": topic, "relevance": 0.98},
                    {"keyword": "Prompt Engineering", "relevance": 0.92},
                    {"keyword": "System Architecture", "relevance": 0.86},
                    {"keyword": "Text Analysis", "relevance": 0.78},
                    {"keyword": "Optimization", "relevance": 0.74}
                ]
            }, indent=2)

        else:
            if temperature > 0.8:
                return f"[High Creativity Mode (Temp: {temperature}, Top-P: {top_p})]: Exploring diverse and imaginative perspectives on {topic}."
            elif temperature < 0.3:
                return f"[Deterministic Mode (Temp: {temperature}, Top-P: {top_p})]: Formal, structured summary of {topic} adhering strictly to defined constraints."
            else:
                return f"[Balanced Mode (Temp: {temperature}, Top-P: {top_p})]: Balanced overview of {topic} synthesizing analytical rigor with creative clarity."
