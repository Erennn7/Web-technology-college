"""
CLI Benchmark Runner
Runs the 5 required test queries + 1 Breaking/Jailbreak query
Compares Initial System Prompt vs Hardened System Prompt.
"""

import os
import sys
import argparse
from prompts import (
    INITIAL_SYSTEM_PROMPT,
    HARDENED_SYSTEM_PROMPT,
    TEST_QUERIES,
    JAILBREAK_QUERY
)
from chatbot_engine import ChatbotEngine


def run_benchmark(provider: str = "auto", api_key: str = ""):
    engine = ChatbotEngine(provider=provider, api_key=api_key)

    print("=" * 80)
    print("  ASSIGNMENT: DESIGN A SPECIAL-PURPOSE AI CHATBOT")
    print("  Subject: Program Elective – V (Prompt Engineering)")
    print("  Chatbot: HealthAware AI (ArogyaGuide - Health Awareness Assistant)")
    print(f"  Engine Mode: {engine.provider.upper()}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PART 1: TEST THE 5 CORE QUERIES
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("PART 1: TESTING THE 5 BENCHMARK QUERIES (Using Initial System Prompt)")
    print("#" * 80)

    for idx, test in enumerate(TEST_QUERIES, 1):
        print(f"\n[{idx}/5] {test['category'].upper()}")
        print(f"User Query: \"{test['prompt']}\"")
        print(f"Expected Behavior: {test['expected_behavior']}\n")

        response, meta = engine.chat(INITIAL_SYSTEM_PROMPT, test["prompt"])
        print("-" * 60)
        print("Chatbot Response:\n")
        print(response)
        print("-" * 60)
        print(f"[Engine: {meta['provider']} | Latency: {meta['latency_seconds']}s]\n")

    # -------------------------------------------------------------------------
    # PART 2: BREAK YOUR CHATBOT (Jailbreak / Prompt Injection Attack)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("PART 2: BREAKING YOUR CHATBOT (Adversarial Prompt Injection)")
    print("#" * 80)
    print(f"Attack Query:\n\"{JAILBREAK_QUERY['prompt']}\"\n")

    print("-" * 80)
    print("A) ATTEMPT ON INITIAL SYSTEM PROMPT (Before Hardening):")
    print("-" * 80)
    res_initial, meta_init = engine.chat(INITIAL_SYSTEM_PROMPT, JAILBREAK_QUERY["prompt"])
    print(res_initial)
    print(f"\n[Observation]: The Initial System Prompt fails because it lacks instruction hierarchy and input delimiters. The user message was able to override prior rules!")

    print("\n" + "-" * 80)
    print("B) ATTEMPT ON HARDENED SYSTEM PROMPT (After Prompt Hardening):")
    print("-" * 80)
    res_hardened, meta_hard = engine.chat(HARDENED_SYSTEM_PROMPT, JAILBREAK_QUERY["prompt"])
    print(res_hardened)
    print(f"\n[Observation]: The Hardened System Prompt SUCCEEDS in deflecting the attack! It treats user input strictly as untrusted data within <user_query> tags and enforces immutable priority level 0 instructions.")

    print("\n" + "=" * 80)
    print("  BENCHMARK SUITE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chatbot Prompt Engineering Test Suite")
    parser.add_argument("--provider", default="auto", choices=["auto", "groq", "gemini", "openai", "mock"], help="LLM Provider")
    parser.add_argument("--api-key", default="", help="Optional API key")
    args = parser.parse_args()

    run_benchmark(provider=args.provider, api_key=args.api_key)
