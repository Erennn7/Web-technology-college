"""
FastAPI Server for AI Content Creation & Analysis System
Serves light-themed minimal Web UI and REST API.
"""

import os
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any

from prompts import (
    STORY_PROMPT,
    POEM_PROMPT,
    SOCIAL_MEDIA_PROMPT,
    PODCAST_PROMPT,
    TEXT_ANALYSIS_PROMPT,
    EXPERIMENTATION_PROMPT,
    build_full_prompt
)
from llm_handler import LLMHandler

app = FastAPI(title="AI Content Creation & Analysis System")

# Absolute path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def read_root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)

# Data Models
class GenerateRequest(BaseModel):
    topic: str
    content_type: str  # story, poem, social, podcast
    provider: Optional[str] = "auto"
    api_key: Optional[str] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95

class AnalyzeRequest(BaseModel):
    text: str
    provider: Optional[str] = "auto"
    api_key: Optional[str] = None

class ExperimentRequest(BaseModel):
    prompt: str
    provider: Optional[str] = "auto"
    api_key: Optional[str] = None
    t1: float = 0.1
    p1: float = 0.5
    t2: float = 0.7
    p2: float = 0.9
    t3: float = 1.2
    p3: float = 0.98


@app.get("/api/prompts")
def get_prompts():
    sample_topic = "Artificial Intelligence in Renewable Energy"
    return {
        "story": {
            "config": STORY_PROMPT,
            "compiled": build_full_prompt(STORY_PROMPT, sample_topic)
        },
        "poem": {
            "config": POEM_PROMPT,
            "compiled": build_full_prompt(POEM_PROMPT, sample_topic)
        },
        "social": {
            "config": SOCIAL_MEDIA_PROMPT,
            "compiled": build_full_prompt(SOCIAL_MEDIA_PROMPT, sample_topic)
        },
        "podcast": {
            "config": PODCAST_PROMPT,
            "compiled": build_full_prompt(PODCAST_PROMPT, sample_topic)
        },
        "analysis": {
            "config": TEXT_ANALYSIS_PROMPT,
            "compiled": build_full_prompt(TEXT_ANALYSIS_PROMPT, sample_topic)
        }
    }


@app.post("/api/generate")
def generate_content(req: GenerateRequest):
    handler = LLMHandler(provider=req.provider, api_key=req.api_key)
    
    if req.content_type == "story":
        prompt_cfg = STORY_PROMPT
    elif req.content_type == "poem":
        prompt_cfg = POEM_PROMPT
    elif req.content_type == "social":
        prompt_cfg = SOCIAL_MEDIA_PROMPT
    elif req.content_type == "podcast":
        prompt_cfg = PODCAST_PROMPT
    else:
        raise HTTPException(status_code=400, detail="Invalid content_type")

    compiled_prompt = build_full_prompt(prompt_cfg, req.topic)
    result_text, meta = handler.generate(compiled_prompt, temperature=req.temperature, top_p=req.top_p)
    return {"text": result_text, "meta": meta}


@app.post("/api/analyze")
def analyze_text(req: AnalyzeRequest):
    handler = LLMHandler(provider=req.provider, api_key=req.api_key)
    compiled_prompt = build_full_prompt(TEXT_ANALYSIS_PROMPT, req.text)
    raw_res, meta = handler.generate(compiled_prompt, temperature=0.2, top_p=0.8)

    try:
        clean_str = raw_res.strip()
        if "```json" in clean_str:
            clean_str = clean_str.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_str:
            clean_str = clean_str.split("```")[1].split("```")[0].strip()
            
        parsed = json.loads(clean_str)
        return {"data": parsed, "meta": meta, "raw": raw_res}
    except Exception:
        return {"data": None, "meta": meta, "raw": raw_res}


@app.post("/api/experiment")
def experiment_parameters(req: ExperimentRequest):
    handler = LLMHandler(provider=req.provider, api_key=req.api_key)
    compiled_prompt = build_full_prompt(EXPERIMENTATION_PROMPT, req.prompt)

    r1, m1 = handler.generate(compiled_prompt, temperature=req.t1, top_p=req.p1)
    r2, m2 = handler.generate(compiled_prompt, temperature=req.t2, top_p=req.p2)
    r3, m3 = handler.generate(compiled_prompt, temperature=req.t3, top_p=req.p3)

    return {
        "profile1": {"text": r1, "meta": m1, "t": req.t1, "p": req.p1},
        "profile2": {"text": r2, "meta": m2, "t": req.t2, "p": req.p2},
        "profile3": {"text": r3, "meta": m3, "t": req.t3, "p": req.p3}
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
