"""
Command-Line Interface (CLI)
AI Content Creation & Analysis System

Run with: python3 cli_main.py --topic "Artificial Intelligence in Healthcare"
"""

import sys
import os
import json
import argparse
from typing import Dict, Any

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


def run_assignment8_cli(topic: str, custom_text: str, provider: str, api_key: str):
    print("=" * 80)
    print("  AI Content Creation & Analysis System")
    print("=" * 80)
    print(f"\n[+] Selected Topic: '{topic}'")
    print(f"[+] Provider Mode: '{provider}'\n")

    handler = LLMHandler(provider=provider, api_key=api_key)

    # ---------------------------------------------------------------------------
    # MODULE 1: PROMPT DESIGN DEMONSTRATION
    # ---------------------------------------------------------------------------
    print("-" * 80)
    print("MODULE 1: PROMPT DESIGN (Role, Context, Constraints, Output Format)")
    print("-" * 80)
    sample_prompt = build_full_prompt(STORY_PROMPT, topic)
    print("Structured Prompt Architecture Example (Story Prompt):\n")
    print(sample_prompt)
    print("-" * 80)

    # ---------------------------------------------------------------------------
    # MODULE 2: CONTENT GENERATION
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODULE 2: CONTENT GENERATION (Story, Poem, Social Media Posts)")
    print("=" * 80)

    print("\n--- 2A. AI Story Generation ---")
    story_prompt = build_full_prompt(STORY_PROMPT, topic)
    story_res, story_meta = handler.generate(story_prompt, temperature=0.8, top_p=0.95)
    print(story_res)
    print(f"\n[Metadata]: Provider={story_meta['provider']} | Latency={story_meta['latency_seconds']}s")

    print("\n--- 2B. AI Poem Generation ---")
    poem_prompt = build_full_prompt(POEM_PROMPT, topic)
    poem_res, poem_meta = handler.generate(poem_prompt, temperature=0.7, top_p=0.9)
    print(poem_res)
    print(f"\n[Metadata]: Provider={poem_meta['provider']} | Latency={poem_meta['latency_seconds']}s")

    print("\n--- 2C. Social Media Campaign Generation ---")
    sm_prompt = build_full_prompt(SOCIAL_MEDIA_PROMPT, topic)
    sm_res, sm_meta = handler.generate(sm_prompt, temperature=0.7, top_p=0.95)
    print(sm_res)
    print(f"\n[Metadata]: Provider={sm_meta['provider']} | Latency={sm_meta['latency_seconds']}s")

    # ---------------------------------------------------------------------------
    # MODULE 3: PODCAST PLANNING
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODULE 3: PODCAST PLANNING (Title, Description, Guest Type, Questions)")
    print("=" * 80)
    pod_prompt = build_full_prompt(PODCAST_PROMPT, topic)
    pod_res, pod_meta = handler.generate(pod_prompt, temperature=0.7, top_p=0.95)
    print(pod_res)
    print(f"\n[Metadata]: Provider={pod_meta['provider']} | Latency={pod_meta['latency_seconds']}s")

    # ---------------------------------------------------------------------------
    # MODULE 4: TEXT ANALYSIS (Sentiment & Keyword Extraction)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODULE 4: TEXT ANALYSIS (Sentiment Analysis & Keyword Extraction)")
    print("=" * 80)
    analysis_input = custom_text or f"Prompt Engineering and Artificial Intelligence have transformed modern software development. While initial setup requires careful context design, the results yield incredible productivity gains, high satisfaction, and inspiring creative breakthroughs!"
    print(f"Input Text for Analysis:\n\"{analysis_input}\"\n")

    analysis_prompt = build_full_prompt(TEXT_ANALYSIS_PROMPT, analysis_input)
    analysis_res, analysis_meta = handler.generate(analysis_prompt, temperature=0.2, top_p=0.8)
    print("LLM JSON Analysis Result:")
    print(analysis_res)
    print(f"\n[Metadata]: Provider={analysis_meta['provider']} | Latency={analysis_meta['latency_seconds']}s")

    # ---------------------------------------------------------------------------
    # MODULE 5: PARAMETER EXPERIMENTATION (Temperature & Top-P)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODULE 5: PARAMETER EXPERIMENTATION (Comparing Temperature & Top-P)")
    print("=" * 80)

    exp_prompt = f"Write a 3-sentence creative perspective on: {topic}"

    params_to_test = [
        {"name": "Deterministic / Focused", "temp": 0.1, "top_p": 0.5},
        {"name": "Balanced Creativity", "temp": 0.7, "top_p": 0.9},
        {"name": "High Creativity / Randomness", "temp": 1.2, "top_p": 0.98}
    ]

    for p in params_to_test:
        print(f"\n--- Setting: {p['name']} (Temperature={p['temp']}, Top-P={p['top_p']}) ---")
        exp_res, exp_meta = handler.generate(exp_prompt, temperature=p["temp"], top_p=p["top_p"])
        print(exp_res)
        print(f"[Latency: {exp_meta['latency_seconds']}s]")

    print("\n" + "=" * 80)
    print("  EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Content Creation & Analysis System")
    parser.add_argument("--topic", type=str, default="Generative AI and Prompt Engineering", help="Topic for content generation & podcast planning")
    parser.add_argument("--text", type=str, default="", help="Custom text for analysis")
    parser.add_argument("--provider", type=str, default="auto", choices=["auto", "gemini", "openai", "mock"], help="LLM Provider")
    parser.add_argument("--api-key", type=str, default="", help="API Key (optional)")

    args = parser.parse_args()
    run_assignment8_cli(args.topic, args.text, args.provider, args.api_key)
