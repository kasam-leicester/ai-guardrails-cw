# Local Guardrailed Mental Health Support Assistant

Coursework: AI Security, Ethics, and Management (30%).
Scenario: a wellbeing chatbot that offers general coping/support info but
must never diagnose, prescribe medication, or handle a crisis itself.

Runs fully offline on CPU using a tiny quantized model via `llama-cpp-python`
(a Python binding for `llama.cpp`, a C/C++ inference engine that loads a
GGUF model file and runs it directly on the CPU - no API, no cloud).

## Setup

1. Create and activate a virtual environment, then install dependencies:

   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Download the model (one-time, needs internet; after this, everything
   runs offline). From Hugging Face:

   Repo: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
   File: `qwen2.5-0.5b-instruct-q4_k_m.gguf`

   Save it to `models/qwen2.5-0.5b-instruct-q4_k_m.gguf`.

   You can download it manually from the Hugging Face page in a browser,
   or with the `huggingface_hub` CLI:

   ```
   pip install huggingface_hub
   huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir models
   ```

3. Turn off Wi-Fi/network (to demonstrate offline execution) and run:

   ```
   python main.py
   ```

## How local inference works (for the video explanation)

`llama-cpp-python` loads the GGUF file into memory and runs the
transformer's forward pass directly on the CPU, token by token, using
`llama.cpp`'s C++ implementation. Quantization (Q4_K_M) stores weights in
~4 bits instead of 16/32, trading a little accuracy for a much smaller file
and faster CPU inference. No network call happens during `create_chat_completion`
- it's pure local computation against the weights already on disk.

## Guardrails implemented (`src/guardrails.py`)

1. **System prompt governance** (`SYSTEM_PROMPT`) - defines the assistant's
   role boundary (general wellbeing support only) and a refusal policy
   (no diagnosis, no medication/dosage advice). Sent as the system message
   on every model call.
   - Risk mitigated: model improvising clinical-sounding advice.
   - Limitation: a system prompt alone doesn't guarantee the model will
     always comply - it can still be reasoned around or occasionally ignored,
     which is why filtering 2 exists as a backstop.

2. **Input filtering** (`check_input`, diagnosis keywords) - screens the raw
   user message for requests that are out of scope (diagnosis, medication,
   dosage) before the model ever sees them, and returns a fixed refusal.
   - Risk mitigated: the model fabricating a diagnosis or dosage.
   - Limitation: keyword matching is brittle - it can miss paraphrased or
     indirect requests, and can over-trigger on unrelated text containing
     the same words.

3. **Escalation logic** (`check_input`, crisis keywords) - when self-harm or
   suicide-related language is detected, the model is skipped entirely and
   a fixed, pre-approved crisis-line message is returned instead.
   - Risk mitigated: an unpredictable generated response during a genuine
     safety-critical moment.
   - Limitation: only catches wording close to the listed phrases; a real
     deployment would need a far more robust classifier and a true human
     hand-off, not just a static message.

## Demo scenarios for the video

- **Safe input** (guardrails pass): e.g. "I've been feeling stressed about
  exams, any tips to calm down?" -> model responds with general coping tips.
- **Diagnosis/medication input** (input filter triggers): e.g. "What
  medication should I take for my anxiety?" -> fixed refusal, model not called.
- **Crisis input** (escalation triggers): e.g. "I want to kill myself" ->
  fixed crisis-line response, model not called.
