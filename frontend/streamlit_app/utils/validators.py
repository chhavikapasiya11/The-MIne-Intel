"""Input validation helpers."""

from __future__ import annotations

from typing import Dict, List, Tuple


FIELD_RANGES = {
    "CMRR": (0, 100),
    "PRSUP": (0, 100),
    "depth_of_ cover": (0, 1000),
    "intersection_diagonal": (0, 20),
    "mining_hight": (0, 10),
}


def validate_payload(payload: Dict[str, float]) -> Tuple[bool, List[str]]:
    """Validate that payload values fall within expected ranges."""
    errors: List[str] = []
    for field, (lower, upper) in FIELD_RANGES.items():
        value = payload.get(field)
        if value is None:
            errors.append(f"`{field}` is required.")
            continue
        if not isinstance(value, (int, float)):
            errors.append(f"`{field}` must be a number.")
            continue
        if value < lower or value > upper:
            errors.append(
                f"`{field}` must be between {lower} and {upper}. Received {value}."
            )
    return len(errors) == 0, errors

