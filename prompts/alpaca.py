"""Stanford Alpaca prompt loader."""

from __future__ import annotations

from typing import List


def load_alpaca_prompts(n: int) -> List[str]:
    """Load the first ``n`` instruction strings from tatsu-lab/alpaca.

    Downloads automatically via the HuggingFace ``datasets`` library. Rows with a
    non-empty ``input`` field have it appended after the instruction.
    """
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    prompts: List[str] = []
    for row in ds:
        if len(prompts) >= n:
            break
        instruction = row["instruction"].strip()
        inp = row.get("input", "").strip()
        prompt = f"{instruction}\n\n{inp}" if inp else instruction
        if prompt:
            prompts.append(prompt)
    return prompts
