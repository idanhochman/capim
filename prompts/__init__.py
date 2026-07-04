"""Prompt loaders for trace collection: Alpaca, GSM8K, and fixed sanity prompts."""

from __future__ import annotations

from typing import List

from prompts.alpaca import load_alpaca_prompts
from prompts.gsm8k import load_gsm8k_prompts
from prompts.vicuna import SANITY_PROMPTS, format_vicuna_prompt

LOADERS = {
    "alpaca": load_alpaca_prompts,
    "gsm8k": load_gsm8k_prompts,
}


def load_prompts(dataset: str, n: int) -> List[str]:
    if dataset not in LOADERS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(LOADERS)}")
    return LOADERS[dataset](n)


__all__ = [
    "load_alpaca_prompts",
    "load_gsm8k_prompts",
    "load_prompts",
    "format_vicuna_prompt",
    "SANITY_PROMPTS",
]
