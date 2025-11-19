"""HTTP client helpers for interacting with the Flask backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import requests

import config


class APIClientError(RuntimeError):
    """Raised when the backend call fails."""


@dataclass(frozen=True)
class PredictionResponse:
    prediction: float


def _build_url(path: str) -> str:
    return f"{config.API_BASE_URL.rstrip('/')}{path}"


def predict_roof_fall_rate(payload: Dict[str, Any]) -> PredictionResponse:
    """Call the Flask `/predict` endpoint and return structured data."""
    url = _build_url(config.PREDICT_ENDPOINT)
    try:
        response = requests.post(url, json=payload, timeout=10)
    except requests.RequestException as exc:  # pragma: no cover - UI handles errors
        raise APIClientError(f"Could not reach backend API: {exc}") from exc

    if response.status_code != 200:
        raise APIClientError(
            f"Backend responded with {response.status_code}: {response.text}"
        )

    data = response.json()
    if "prediction" not in data:
        raise APIClientError("Backend response is missing the `prediction` field.")

    return PredictionResponse(prediction=float(data["prediction"]))

