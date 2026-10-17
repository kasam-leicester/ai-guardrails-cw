"""Guardrail logic for the local mental health support assistant.

Three guardrails, applied in order, before any text reaches the model:
1. System prompt governance  -> SYSTEM_PROMPT (role + refusal policy)
2. Input filtering           -> check_input() keyword/intent screening
3. Escalation logic          -> check_input() short-circuits to a fixed
                                 crisis response instead of calling the model
"""

# --- Guardrail 1: System prompt governance -------------------------------
# Defines the assistant's role boundary and refusal policy. Sent as the
# "system" message on every call so the model is steered away from acting
# like a clinician, even for inputs that pass the filters below.
SYSTEM_PROMPT = """\
<role>
You are a general wellbeing support assistant. You are NOT a doctor,
therapist, or emergency service.
</role>

<allowed>
You may share general coping strategies, breathing/grounding techniques,
and encouragement to speak with a qualified professional.
</allowed>

<refusals>
You must refuse to:
- diagnose any condition
- recommend or name any medication or dosage
- give any opinion on whether someone's symptoms indicate a specific illness
If asked to do any of these, politely refuse and suggest the person
consult a licensed professional instead.
</refusals>

<style>
Keep answers short and supportive.
</style>
"""

# --- Guardrail 2: Input filtering -----------------------------------------
# Risky-intent keywords: requests for a diagnosis, medication, or dosage.
# These are not emergencies, but they are outside the assistant's role, so
# they are flagged and handled with a fixed refusal rather than trusting the
# model to refuse on its own every time.
DIAGNOSIS_KEYWORDS = [
    "diagnose", "diagnosis", "what medication", "which medication",
    "what dose", "what dosage", "how many mg", "prescribe", "prescription",
    "do i have depression", "do i have anxiety", "do i have bipolar",
]

# --- Guardrail 3: Escalation logic -----------------------------------------
# Crisis keywords: when matched, the model is never called. The system
# returns a fixed, pre-approved escalation message pointing to human help.
# This removes the risk of an unpredictable generated response during a
# genuine safety-critical moment.
CRISIS_KEYWORDS = [
    "kill myself", "suicide", "end my life", "want to die",
    "self harm", "self-harm", "hurt myself", "no reason to live",
    "overdose on purpose",
]

CRISIS_RESPONSE = (
    "I'm really sorry you're feeling this way. I'm not able to help with "
    "this myself, so please reach out to a real person right now: "
    "call 988 (Suicide & Crisis Lifeline, US) or 999/111 (UK), or contact "
    "emergency services. If you can, please also tell someone you trust "
    "what you just told me."
)

DIAGNOSIS_REFUSAL_RESPONSE = (
    "I can't diagnose conditions or recommend medications or dosages - "
    "that needs a licensed doctor or pharmacist. I can share general, "
    "non-medical coping and wellbeing information if that would help."
)


def check_input(user_text: str):
    """Screen user input before it reaches the model.

    Returns a tuple (status, response_or_none):
      - ("crisis", CRISIS_RESPONSE)            escalation guardrail triggered
      - ("diagnosis", DIAGNOSIS_REFUSAL_RESPONSE)  input filter triggered
      - ("ok", None)                            safe to send to the model
    """
    text = user_text.lower()

    for phrase in CRISIS_KEYWORDS:
        if phrase in text:
            return "crisis", CRISIS_RESPONSE

    for phrase in DIAGNOSIS_KEYWORDS:
        if phrase in text:
            return "diagnosis", DIAGNOSIS_REFUSAL_RESPONSE

    return "ok", None
