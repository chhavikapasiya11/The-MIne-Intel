"""Simple natural language extraction for mine telemetry fields."""

from __future__ import annotations

import re
from typing import Dict, Optional


FIELD_SYNONYMS = {
    "CMRR": ["cmrr", "roof rating"],
    "PRSUP": ["prsup", "support load", "roof support"],
    "depth_of_ cover": ["depth of cover", "overburden", "cover depth"],
    "intersection_diagonal": ["intersection diagonal", "diagonal", "crosscut diagonal"],
    "mining_hight": ["mining height", "seam height", "extraction height"],
}


def _extract_number(snippet: str) -> Optional[float]:
    match = re.search(r"(-?\d+(?:\.\d+)?)", snippet)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if "percent" in snippet or "%" in snippet:
        return value
    return value


def extract_features_from_text(text: str) -> Dict[str, Optional[float]]:
    """Return best-effort field extraction from free-form text."""
    lowered = text.lower()
    extracted: Dict[str, Optional[float]] = {field: None for field in FIELD_SYNONYMS}

    for field, synonyms in FIELD_SYNONYMS.items():
        for synonym in synonyms:
            escaped = re.escape(synonym)
            pattern = (
                rf"{escaped}[^0-9-]*(-?\d+(?:\.\d+)?)"
                rf"(?:\s?(?:meters|meter|m|percent|%))?"
            )
            match = re.search(pattern, lowered)
            if match:
                extracted[field] = float(match.group(1))
                break
        if extracted[field] is not None:
            continue

        # Support "value for synonym" phrasing, e.g., "set CMRR to 45"
        alt_pattern = rf"(?:set|make|is|at)\s+{escaped}[^0-9-]*(-?\d+(?:\.\d+)?)"
        match = re.search(alt_pattern, lowered)
        if match:
            extracted[field] = float(match.group(1))

    return extracted

