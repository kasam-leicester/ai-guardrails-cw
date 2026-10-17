"""Thin wrapper around llama-cpp-python for local, offline inference."""

from llama_cpp import Llama

from .guardrails import SYSTEM_PROMPT


def load_model(model_path: str) -> Llama:
    """Load a local GGUF model fully offline via llama.cpp.

    n_ctx is the context window (tokens); n_threads controls CPU
    parallelism for inference. No network access happens here or during
    generation - llama.cpp reads the GGUF file straight off disk.
    """
    return Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        verbose=False,
    )


def generate_response(llm: Llama, user_text: str) -> str:
    """Send the system prompt + user message to the model and return text."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    result = llm.create_chat_completion(
        messages=messages,
        max_tokens=350,
        temperature=0.7,
    )
    return result["choices"][0]["message"]["content"].strip()
