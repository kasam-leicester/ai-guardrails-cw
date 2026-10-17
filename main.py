"""CLI entry point for the local, guardrailed mental health support assistant.

Scenario: a wellbeing chatbot that must never diagnose, prescribe, or
handle a mental health crisis on its own. Everything below runs fully
offline once the GGUF model file has been downloaded - no API calls,
no cloud service, no internet connection required at run time.
"""

import os
import sys

from src.assistant import load_model, generate_response
from src.guardrails import check_input

MODEL_PATH = os.path.join("models", "qwen2.5-0.5b-instruct-q4_k_m.gguf")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}")
        print("Download it first - see README.md for instructions.")
        sys.exit(1)

    print("Loading local model (fully offline)...")
    llm = load_model(MODEL_PATH)
    print("Model loaded. Type 'quit' to exit.\n")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in ("quit", "exit"):
            break
        if not user_text:
            continue

        status, fixed_response = check_input(user_text)

        if status == "crisis":
            print(f"[ESCALATION GUARDRAIL TRIGGERED]\nAssistant: {fixed_response}\n")
            continue

        if status == "diagnosis":
            print(f"[INPUT FILTER GUARDRAIL TRIGGERED]\nAssistant: {fixed_response}\n")
            continue

        response = generate_response(llm, user_text)
        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    main()
