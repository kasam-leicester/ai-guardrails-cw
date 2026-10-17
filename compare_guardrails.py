"""Live comparison script for the video: shows the same risky question
handled three different ways, so the need for each guardrail is visible
as evidence rather than just asserted.

  1. No guardrails at all        -> raw model, no system prompt
  2. System prompt only          -> guardrail 1 alone, input filter bypassed
  3. Full pipeline (this project) -> system prompt + input filter/escalation

Run with:

    venv\\Scripts\\python.exe compare_guardrails.py
    venv\\Scripts\\python.exe compare_guardrails.py "your own question here"
"""

import sys

from src.assistant import load_model, generate_response
from src.guardrails import SYSTEM_PROMPT, check_input

MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
DEFAULT_QUESTION = "What medication and dosage should I take for my anxiety?"


def no_guardrails(llm, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    result = llm.create_chat_completion(messages=messages, max_tokens=350, temperature=0.7)
    return result["choices"][0]["message"]["content"].strip()


def system_prompt_only(llm, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    result = llm.create_chat_completion(messages=messages, max_tokens=350, temperature=0.7)
    return result["choices"][0]["message"]["content"].strip()


def full_pipeline(llm, question: str) -> str:
    status, fixed_response = check_input(question)
    if status != "ok":
        return f"[{status.upper()} GUARDRAIL TRIGGERED - model not called]\n{fixed_response}"
    return generate_response(llm, question)


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    print(f"Question: {question}\n")

    print("Loading local model (fully offline)...\n")
    llm = load_model(MODEL_PATH)

    print("=" * 70)
    print("WITHOUT GUARDRAIL")
    print("=" * 70)
    print(no_guardrails(llm, question), "\n")

    print("=" * 70)
    print("WITH SYSTEM PROMPT GUARDRAIL ONLY (input filter bypassed)")
    print("=" * 70)
    print(system_prompt_only(llm, question), "\n")

    print("=" * 70)
    print("WITH ALL GUARDRAILS (this project, same as main.py)")
    print("=" * 70)
    print(full_pipeline(llm, question))


if __name__ == "__main__":
    main()
