"""
LLM Interface Handler
Supports:
1. Google Gemini API (via google-genai package & direct HTTP REST fallback)
2. OpenAI API (via openai package & direct HTTP REST fallback)
3. Offline Simulation Engine (if no API key provided)
"""

import os
import json
import time
import re
import urllib.request
import urllib.error
from typing import Dict, Any, Tuple, Optional

# Check optional package imports
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
        self.raw_provider = provider.lower()
        self.api_key = (api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
        
        # Auto-detect provider if API key is present
        if self.raw_provider == "auto":
            if self.api_key:
                if self.api_key.startswith("sk-"):
                    self.provider = "openai"
                else:
                    self.provider = "gemini"
            else:
                self.provider = "mock"
        else:
            self.provider = self.raw_provider

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        model_name: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        start_time = time.time()

        # If no key or mock requested
        if not self.api_key or self.provider == "mock":
            res_text = self._mock_generation(prompt, temperature, top_p)
            elapsed = round(time.time() - start_time, 3)
            return res_text, {
                "provider": "Offline Engine (No API Key)",
                "model": "rule-engine-v1",
                "temperature": temperature,
                "top_p": top_p,
                "latency_seconds": elapsed,
                "status": "Success"
            }

        # Handle Gemini Provider
        if self.provider == "gemini":
            # Try SDK first if available
            if HAS_GENAI:
                try:
                    client = genai.Client(api_key=self.api_key)
                    selected_model = model_name or "gemini-2.0-flash"
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
                    return res.text, {
                        "provider": "Google Gemini API (SDK)",
                        "model": selected_model,
                        "temperature": temperature,
                        "top_p": top_p,
                        "latency_seconds": elapsed,
                        "status": "Success"
                    }
                except Exception as sdk_err:
                    print(f"[LLMHandler] SDK error: {sdk_err}. Trying direct REST API fallback...")

            # Direct HTTP REST API call (Works everywhere without extra packages)
            try:
                res_text, model_used = self._call_gemini_rest(prompt, self.api_key, temperature, top_p, model_name)
                elapsed = round(time.time() - start_time, 3)
                return res_text, {
                    "provider": "Google Gemini API (REST)",
                    "model": model_used,
                    "temperature": temperature,
                    "top_p": top_p,
                    "latency_seconds": elapsed,
                    "status": "Success"
                }
            except Exception as rest_err:
                error_msg = f"Gemini API Error: {str(rest_err)}"
                print(f"[LLMHandler] REST API error: {rest_err}")
                res_text = self._mock_generation(prompt, temperature, top_p)
                elapsed = round(time.time() - start_time, 3)
                return res_text, {
                    "provider": f"Offline Fallback ({error_msg[:60]}...)",
                    "model": "mock-fallback",
                    "temperature": temperature,
                    "top_p": top_p,
                    "latency_seconds": elapsed,
                    "status": error_msg
                }

        # Handle OpenAI Provider
        if self.provider == "openai":
            if HAS_OPENAI:
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
                    return content, {
                        "provider": "OpenAI API",
                        "model": selected_model,
                        "temperature": temperature,
                        "top_p": top_p,
                        "latency_seconds": elapsed,
                        "status": "Success"
                    }
                except Exception as oai_err:
                    error_msg = f"OpenAI API Error: {str(oai_err)}"
                    res_text = self._mock_generation(prompt, temperature, top_p)
                    elapsed = round(time.time() - start_time, 3)
                    return res_text, {
                        "provider": f"Offline Fallback ({error_msg[:60]}...)",
                        "model": "mock-fallback",
                        "temperature": temperature,
                        "top_p": top_p,
                        "latency_seconds": elapsed,
                        "status": error_msg
                    }

        # Default fallback
        res_text = self._mock_generation(prompt, temperature, top_p)
        elapsed = round(time.time() - start_time, 3)
        return res_text, {
            "provider": "Offline Engine",
            "model": "mock",
            "temperature": temperature,
            "top_p": top_p,
            "latency_seconds": elapsed,
            "status": "Success"
        }

    def _call_gemini_rest(
        self, prompt: str, api_key: str, temperature: float, top_p: float, model_name: Optional[str]
    ) -> Tuple[str, str]:
        """Direct zero-dependency Google Gemini REST API caller."""
        models_to_try = [model_name] if model_name else ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        
        last_error = None
        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            payload = {
                "contents": [
                    {
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": float(temperature),
                    "topP": float(top_p)
                }
            }

            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    res_body = json.loads(response.read().decode("utf-8"))
                    text = res_body["candidates"][0]["content"]["parts"][0]["text"]
                    return text, model
            except urllib.error.HTTPError as e:
                err_resp = e.read().decode("utf-8") if e.fp else str(e)
                last_error = f"HTTP {e.code}: {err_resp}"
            except Exception as e:
                last_error = str(e)

        raise RuntimeError(last_error or "Gemini REST API Call Failed")

    def _mock_generation(self, prompt: str, temperature: float, top_p: float) -> str:
        lower_prompt = prompt.lower()
        topic_match = re.search(r"User Input / Topic:\s*(.*)", prompt, re.IGNORECASE)
        topic = topic_match.group(1).strip() if topic_match else "Current Events and Society"

        clean_topic = topic.strip().capitalize()

        if "story" in lower_prompt:
            return (
                f"Title: Reflections on {clean_topic}\n\n"
                f"Story:\n"
                f"The morning air was thick with anticipation as citizens gathered across the city square. "
                f"For weeks, discussions surrounding {clean_topic.lower()} had dominated conversations in homes, cafes, and public forums. "
                f"Demonstrators stood shoulder to shoulder, raising signs and sharing stories that echoed a deep desire for systemic change and reform.\n\n"
                f"As dusk fell, community leaders stepped forward to lead a peaceful dialogue. Tensions initially flared when differing perspectives met, "
                f"yet the collective commitment to civic engagement helped maintain order. Organizers and representatives met late into the evening to channel "
                f"the movement's momentum into concrete policy proposals and structured negotiations.\n\n"
                f"By midnight, a peaceful assembly concluded with a shared pledge for ongoing dialogue. The events demonstrated how public voice and active "
                f"participation remain vital forces in shaping societal progress and democratic accountability.\n\n"
                f"Moral / Takeaway: Constructive dialogue and peaceful civic expression are fundamental catalysts for meaningful societal progress."
            )
        
        elif "poem" in lower_prompt:
            return (
                f"Title: Voices of {clean_topic}\n\n"
                f"A sudden call across the square,\n"
                f"A unified and steady prayer.\n"
                f"In thoughts of {clean_topic.lower()} we stand,\n"
                f"Seeking hope across the land.\n\n"
                f"Through stormy days and twilight sky,\n"
                f"Resilient voices rising high.\n"
                f"With courageous hearts and open eyes,\n"
                f"A clearer vision starts to rise.\n\n"
                f"Though roads are long and shadows deep,\n"
                f"The promises we strive to keep.\n"
                f"And as the stars light up the night,\n"
                f"Truth illuminates our sight."
            )
            
        elif "social media" in lower_prompt or "twitter" in lower_prompt:
            return (
                f"--- TWITTER / X POST ---\n"
                f"Understanding the broader context around #{clean_topic.replace(' ', '')}. Civic awareness and open dialogue play a key role in modern society. What are your perspectives?\n"
                f"Hashtags: #{clean_topic.replace(' ', '')} #CurrentEvents #Society #Perspective\n\n"
                f"--- LINKEDIN POST ---\n"
                f"Analyzing the Impact of {clean_topic}\n\n"
                f"As events unfold, leaders and analysts are evaluating the socio-economic and policy implications surrounding {clean_topic.lower()}.\n\n"
                f"Key Perspectives:\n"
                f"1. Community engagement & public sentiment\n"
                f"2. Institutional response & policy reform\n"
                f"3. Long-term societal outlook and stability\n\n"
                f"How do you view these developments from a strategic perspective? Share your thoughts below.\n"
                f"Hashtags: #{clean_topic.replace(' ', '')} #PublicPolicy #Leadership #Society\n\n"
                f"--- INSTAGRAM CAPTION ---\n"
                f"Voices, perspective, and community engagement surrounding {clean_topic}. Every movement tells a story of people striving for change.\n\n"
                f"Hashtags: #{clean_topic.replace(' ', '')} #CurrentEvents #Community #Perspective"
            )

        elif "podcast" in lower_prompt or "interview" in lower_prompt:
            return (
                f"Episode Title: Understanding {clean_topic}: Perspectives, Impact, and Policy Outlook\n\n"
                f"Description:\n"
                f"In this episode, we examine the multi-faceted dynamics of {clean_topic.lower()} with leading socio-political and policy analysts. "
                f"We explore historical context, key triggers, public sentiment, and future implications. Designed for listeners seeking balanced, in-depth analysis.\n\n"
                f"Target Guest Profile:\n"
                f"- Ideal Role: Senior Political Analyst / Sociologist\n"
                f"- Expertise: Civic Dynamics, Public Policy, Social Movements\n"
                f"- Rationale: Deep domain expertise in analyzing public movements and policy reform.\n\n"
                f"Interview Questions:\n"
                f"1. Could you provide the historical context leading up to recent developments in {clean_topic.lower()}?\n"
                f"2. What are the key drivers shaping public participation in this movement?\n"
                f"3. How are institutions and policy makers currently responding to these events?\n"
                f"4. What role do digital media and prompt engineering / NLP play in monitoring public sentiment?\n"
                f"5. What long-term transformations do you anticipate emerging from this situation?"
            )

        elif "sentiment" in lower_prompt or "analysis" in lower_prompt or "json" in lower_prompt:
            return json.dumps({
                "sentiment": {
                    "score": 0.15,
                    "label": "Neutral",
                    "explanation": f"The text presents balanced narrative regarding {clean_topic.lower()} with constructive and reflective language."
                },
                "keywords": [
                    {"keyword": clean_topic, "relevance": 0.98},
                    {"keyword": "Public Dialogue", "relevance": 0.90},
                    {"keyword": "Civic Engagement", "relevance": 0.85},
                    {"keyword": "Policy Reform", "relevance": 0.78},
                    {"keyword": "Societal Impact", "relevance": 0.72}
                ]
            }, indent=2)

        else:
            if temperature > 0.8:
                return f"[High Creativity Mode (Temp: {temperature}, Top-P: {top_p})]: Exploring creative and expressive perspectives on {clean_topic}."
            elif temperature < 0.3:
                return f"[Deterministic Mode (Temp: {temperature}, Top-P: {top_p})]: Formal, structured summary of {clean_topic} adhering strictly to constraints."
            else:
                return f"[Balanced Mode (Temp: {temperature}, Top-P: {top_p})]: Balanced overview of {clean_topic} synthesizing analytical context with clear presentation."
