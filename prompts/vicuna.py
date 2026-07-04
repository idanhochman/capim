"""
Prompt formatting for the fixed backbone (Vicuna-7B-v1.3).

Vicuna-7B-v1.3 is the only backbone in scope, so there is exactly one prompt
template to carry.
"""

from __future__ import annotations

VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def format_vicuna_prompt(prompt: str) -> str:
    """Standard Vicuna v1.3 conversation format (FastChat convention)."""
    return f"{VICUNA_SYSTEM}\n\nUSER: {prompt}\nASSISTANT:"


# 5 prompts spanning factual, instruction, math, and creative tasks -- no dataset
# download needed.  Diversity in expected confidence is intentional: factual/
# formulaic prompts should produce high-confidence draft tokens; open-ended
# creative prompts should produce lower-confidence ones.
SANITY_PROMPTS = [
    "List the planets of the solar system in order from the Sun.",
    "Write a Python function that returns the factorial of a non-negative integer n.",
    "A store sells apples for $0.50 each and oranges for $0.75 each. If I buy 3 "
    "apples and 4 oranges, how much do I spend in total?",
    "Write a short story (3-4 sentences) about a robot who discovers it can dream.",
    "Describe the colour blue to someone who has been blind from birth.",
]
