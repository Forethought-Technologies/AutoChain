ESTIMATE_CONFIDENCE_PROMPT = """Given the system policy assistant needs to strictly follow and
the conversation history between user and assistant so far,
"System policy: ${policy}
${conversation_history}"

How confident are you the next step from assistant should be the following:
"${assistant_message}"

Estimate the confidence from 1-5, 1 being the least confident and 5 being the most confident.
Confidence:
"""
