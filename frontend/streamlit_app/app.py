"""Streamlit entry point for the Mine-Intel frontend."""

from __future__ import annotations

import pathlib
from typing import Any, Dict, Optional

import streamlit as st

try:  # pragma: no cover - optional dependency during linting
    from streamlit_mic_recorder import speech_to_text  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    speech_to_text = None

import config
from components.layout import (
    render_error,
    render_page_header,
    render_prediction_card,
    render_sidebar,
)
from services.api_client import (
    APIClientError,
    predict_roof_fall_rate,
)
from utils.nlp import extract_features_from_text
from utils.validators import validate_payload


FEATURE_FIELDS = {
    "CMRR": {
        "help": "Coal Mine Roof Rating (0-100)",
        "min_value": 0.0,
        "max_value": 100.0,
        "value": 50.0,
    },
    "PRSUP": {
        "help": "Percentage of roof support load (0-100)",
        "min_value": 0.0,
        "max_value": 100.0,
        "value": 40.0,
    },
    "depth_of_ cover": {
        "help": "Depth of cover in meters",
        "min_value": 0.0,
        "max_value": 1000.0,
        "value": 200.0,
    },
    "intersection_diagonal": {
        "help": "Intersection diagonal in meters",
        "min_value": 0.0,
        "max_value": 20.0,
        "value": 5.0,
    },
    "mining_hight": {
        "help": "Mining height in meters",
        "min_value": 0.0,
        "max_value": 10.0,
        "value": 2.5,
    },
}


def _load_custom_css() -> None:
    css_path = pathlib.Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text()}</style>",
            unsafe_allow_html=True,
        )


def _build_payload(form_values: Dict[str, float]) -> Dict[str, float]:
    return {field: float(value) for field, value in form_values.items()}


def _seed_inputs_from_nlp(extracted: Dict[str, Optional[float]]) -> None:
    for field, value in extracted.items():
        if value is None:
            continue
        st.session_state[f"field_{field}"] = float(value)


def _initialize_state() -> None:
    if "nlp_values" not in st.session_state:
        st.session_state["nlp_values"] = {}
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {
                "role": "assistant",
                "content": (
                    "Describe the latest mine readings or use the microphone. "
                    "I'll fill in the numeric inputs."
                ),
            }
        ]


def _append_chat(role: str, content: str) -> None:
    st.session_state["chat_history"].append({"role": role, "content": content})
    if len(st.session_state["chat_history"]) > 12:
        st.session_state["chat_history"] = st.session_state["chat_history"][-12:]


def _process_nlp_input(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return "I didn't catch anything. Could you repeat the readings?"

    extracted = extract_features_from_text(cleaned)
    st.session_state["nlp_values"] = {
        field: value for field, value in extracted.items() if value is not None
    }
    _seed_inputs_from_nlp(extracted)

    filled = [
        f"{field}: {value}"
        for field, value in extracted.items()
        if value is not None
    ]
    missing = [field for field, value in extracted.items() if value is None]

    if filled:
        response = "Updated " + ", ".join(filled) + "."
    else:
        response = "I couldn't detect any numbers in that message."

    if missing:
        response += " Please provide " + ", ".join(missing) + "."

    return response


def _extract_transcript(payload: Optional[Any]) -> Optional[str]:
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload.strip() or None
    if isinstance(payload, dict):
        text = payload.get("text") or payload.get("transcript")
        if text and text.strip():
            return text.strip()
    return None


def main() -> None:
    st.set_page_config(
        page_title=config.APP_TITLE,
        page_icon="⛏️",
        layout="centered",
    )
    st.markdown('<div class="mine-intel">', unsafe_allow_html=True)
    _load_custom_css()
    _initialize_state()

    render_sidebar()
    render_page_header()

    form_col, chat_col = st.columns([2, 1], gap="large")
    submitted = False
    form_values: Dict[str, float] = {}

    with form_col:
        with st.form("prediction-form", clear_on_submit=False):
            col1, col2 = st.columns(2)

            for index, (field, options) in enumerate(FEATURE_FIELDS.items()):
                column = col1 if index % 2 == 0 else col2
                input_key = f"field_{field}"
                default_value = st.session_state.get(input_key, options["value"])
                form_values[field] = column.number_input(
                    field,
                    min_value=options["min_value"],
                    max_value=options["max_value"],
                    value=float(default_value),
                    help=options["help"],
                    key=input_key,
                )

            submitted = st.form_submit_button("Predict roof fall rate")

    if submitted:
        payload = _build_payload(form_values)
        valid, errors = validate_payload(payload)
        if not valid:
            render_error(" • ".join(errors))
        else:
            with st.spinner("Contacting backend model..."):
                try:
                    prediction = predict_roof_fall_rate(payload)
                except APIClientError as exc:
                    render_error(str(exc))
                else:
                    render_prediction_card(prediction.prediction)

    with chat_col:
        chat_col.subheader("Voice & chat assistant")
        chat_col.caption("Speak or type natural language. I'll populate the form.")
        chat_col.markdown('<div class="chat-panel">', unsafe_allow_html=True)
        chat_col.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        for message in st.session_state["chat_history"]:
            role = message["role"]
            icon = "👷" if role == "user" else "🤖"
            chat_col.markdown(
                f'<div class="chat-message {role}">{icon} {message["content"]}</div>',
                unsafe_allow_html=True,
            )
        chat_col.markdown("</div>", unsafe_allow_html=True)

        voice_message = None
        if speech_to_text is not None:
            voice_payload = speech_to_text(
                language="en",
                use_container_width=True,
                just_once=True,
                key="voice_input",
                start_prompt="Hold to speak",
                stop_prompt="Processing...",
            )
            voice_message = _extract_transcript(voice_payload)
        else:
            chat_col.info(
                "Install `streamlit-mic-recorder` to enable voice capture.",
                icon="🎙️",
            )

        if voice_message:
            _append_chat("user", voice_message)
            response = _process_nlp_input(voice_message)
            _append_chat("assistant", response)

        prompt = chat_col.chat_input("Describe the readings")
        if prompt:
            _append_chat("user", prompt)
            response = _process_nlp_input(prompt)
            _append_chat("assistant", response)

        chat_col.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

