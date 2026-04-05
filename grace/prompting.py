from __future__ import annotations

import re
from typing import Any


PROMPT_TEMPLATES = {
    "basic": (
        "In the above code snippet, check for potential security vulnerabilities "
        "and output strictly '1' for Vulnerable or '0' for Non-vulnerable."
    ),
    "identity": "You are now an excellent programmer.",
    "domain": "You are conducting a function vulnerability detection task for C/C++ language.",
    "nodes": "The node information of the function is as follows:",
    "edges": "The edge information of the function is as follows:",
    "example": "Here is an example for you to learn from:",
    "example_label": "The example label is:",
}


def build_prompt(record: dict[str, Any], mode: str = "grace") -> str:
    func = (record.get("func") or "")[:4000]
    parts = [
        func,
        PROMPT_TEMPLATES["basic"],
        PROMPT_TEMPLATES["identity"],
        PROMPT_TEMPLATES["domain"],
    ]

    if mode == "grace":
        node = (record.get("node") or "")[:2000]
        edge = (record.get("edge") or "")[:2000]
        example = (record.get("example") or "")[:4000]
        example_target = record.get("example_target")
        if node:
            parts.extend([PROMPT_TEMPLATES["nodes"], node])
        if edge:
            parts.extend([PROMPT_TEMPLATES["edges"], edge])
        if example:
            parts.extend([PROMPT_TEMPLATES["example"], example])
            if example_target is not None:
                parts.extend(
                    [
                        PROMPT_TEMPLATES["example_label"],
                        "1" if int(example_target) == 1 else "0",
                    ]
                )

    return "\n\n".join(part for part in parts if part)


def normalize_prediction(prediction: str) -> int | None:
    value = prediction.strip().lower()
    if value in {"1", "vulnerable"}:
        return 1
    if value in {"0", "non-vulnerable", "non_vulnerable", "non vulnerable"}:
        return 0

    match = re.search(r"\b([01])\b", value)
    if match:
        return int(match.group(1))

    if "non-vulnerable" in value or "not vulnerable" in value:
        return 0
    if "vulnerable" in value:
        return 1
    return None

