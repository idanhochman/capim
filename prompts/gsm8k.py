"""GSM8K math word problem loader."""

from __future__ import annotations

from typing import List


def load_gsm8k_prompts(n: int) -> List[str]:
    """Load the first ``n`` questions from GSM8K's ``main`` train split.

    Downloads automatically via the HuggingFace ``datasets`` library.
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    return [row["question"].strip() for row in ds][:n]
