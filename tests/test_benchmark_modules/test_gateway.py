"""Tests for the gateway benchmark module."""

from copy import deepcopy

from euroeval.benchmark_modules.gateway import (
    _normalise_response_text,
    _strip_trailing_prompt_whitespace,
)


def test_strip_trailing_whitespace_on_prompt_fields() -> None:
    batch = {
        "text": ["Grammatisk korrekt: ", "No trailing"],
        "prompt": ["Svar: \t", "Already clean"],
    }

    cleaned = _strip_trailing_prompt_whitespace(deepcopy(batch))

    assert cleaned["text"] == ["Grammatisk korrekt:", "No trailing"]
    assert cleaned["prompt"] == ["Svar:", "Already clean"]


def test_strip_trailing_whitespace_preserves_newlines() -> None:
    batch = {
        "text": ["Line:\n", "Another line:\n \t"],
        "prompt": ["Question?\n", "Trailing newline only\n"],
    }

    cleaned = _strip_trailing_prompt_whitespace(deepcopy(batch))

    assert cleaned["text"] == ["Line:\n", "Another line:\n"]
    assert cleaned["prompt"] == ["Question?\n", "Trailing newline only\n"]


def test_strip_trailing_whitespace_leaves_non_strings() -> None:
    batch = {"text": [None, 1, "Label: "]}

    cleaned = _strip_trailing_prompt_whitespace(deepcopy(batch))

    assert cleaned["text"] == [None, 1, "Label:"]


def test_normalise_response_text_for_structured_outputs() -> None:
    text = "  {\"PER\": [\"Alice\"]}\n"

    assert (
        _normalise_response_text(text, uses_structured_output=True)
        == '{"PER": ["Alice"]}'
    )


def test_normalise_response_text_for_non_structured_outputs() -> None:
    text = '  "Answer"\n'

    assert _normalise_response_text(text, uses_structured_output=False) == "Answer"
