"""Tests for _validate_summary robustness."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.summary.summary_service import (
    _validate_summary,
    NOT_STATED,
    VALID_KEYS,
)

import json


# ---- Case 1: normal JSON with all five keys ----

def test_normal_json_all_keys():
    obj = {
        "objective": "The paper studies X.",
        "main_contribution": "Proposes method Y.",
        "methodology": "Uses dataset Z.",
        "key_findings": "Achieves 95% accuracy.",
        "limitations_conclusion": "Small dataset.",
    }
    result = _validate_summary(json.dumps(obj))
    assert result is not None
    for key in VALID_KEYS:
        assert result[key] == obj[key]
    print("PASS: normal JSON")


# ---- Case 2: JSON inside markdown code fences ----

def test_json_in_code_fences():
    obj = {
        "objective": "Studies X.",
        "main_contribution": "Method Y.",
        "methodology": "Dataset Z.",
        "key_findings": "95% accuracy.",
        "limitations_conclusion": "Small dataset.",
    }
    fenced = f"```json\n{json.dumps(obj, indent=2)}\n```"
    result = _validate_summary(fenced)
    assert result is not None
    for key in VALID_KEYS:
        assert result[key] == obj[key]
    print("PASS: code fences")


# ---- Case 3: nested/stringified JSON (all fields inside one string value) ----

def test_nested_stringified_json():
    """Simulates the exact bug: LLM wraps entire summary as a string in 'objective'."""
    inner = {
        "objective": "Studies X.",
        "main_contribution": "Method Y.",
        "methodology": "Dataset Z.",
        "key_findings": "95% accuracy.",
        "limitations_conclusion": "Small dataset.",
    }
    outer = {"objective": json.dumps(inner)}
    result = _validate_summary(json.dumps(outer))
    assert result is not None
    for key in VALID_KEYS:
        assert result[key] == inner[key]
    print("PASS: nested stringified JSON")


# ---- Case 3b: entire LLM output is a JSON string (not object) ----

def test_entire_output_is_json_string():
    """LLM returns '\"{ ... }\"' — a JSON string wrapping the object."""
    inner = {
        "objective": "Studies X.",
        "main_contribution": "Method Y.",
        "methodology": "Dataset Z.",
        "key_findings": "95% accuracy.",
        "limitations_conclusion": "Small dataset.",
    }
    # The raw output is a JSON-encoded string
    raw = json.dumps(json.dumps(inner))
    result = _validate_summary(raw)
    assert result is not None
    for key in VALID_KEYS:
        assert result[key] == inner[key]
    print("PASS: entire output is JSON string")


# ---- Case 3c: nested inside a wrapper dict ----

def test_wrapped_in_outer_dict():
    inner = {
        "objective": "Studies X.",
        "main_contribution": "Method Y.",
        "methodology": "Dataset Z.",
        "key_findings": "95% accuracy.",
        "limitations_conclusion": "Small dataset.",
    }
    wrapper = {"summary": inner}
    result = _validate_summary(json.dumps(wrapper))
    assert result is not None
    for key in VALID_KEYS:
        assert result[key] == inner[key]
    print("PASS: wrapped in outer dict")


# ---- Case 4: missing key ----

def test_missing_key():
    obj = {
        "objective": "Studies X.",
        "main_contribution": "Method Y.",
        # methodology missing
        "key_findings": "95% accuracy.",
        "limitations_conclusion": "Small dataset.",
    }
    result = _validate_summary(json.dumps(obj))
    assert result is not None
    assert result["objective"] == "Studies X."
    assert result["main_contribution"] == "Method Y."
    assert result["methodology"] == NOT_STATED
    assert result["key_findings"] == "95% accuracy."
    assert result["limitations_conclusion"] == "Small dataset."
    print("PASS: missing key")


# ---- Case 5: invalid JSON ----

def test_invalid_json():
    result = _validate_summary("this is not json at all {{{{")
    assert result is None
    print("PASS: invalid JSON returns None")


# ---- Case 6: empty / blank input ----

def test_empty_input():
    assert _validate_summary("") is None
    assert _validate_summary("   ") is None
    print("PASS: empty input")


# ---- Case 7: non-dict JSON (list) ----

def test_list_json():
    result = _validate_summary('["a", "b"]')
    assert result is None
    print("PASS: list JSON returns None")


if __name__ == "__main__":
    test_normal_json_all_keys()
    test_json_in_code_fences()
    test_nested_stringified_json()
    test_entire_output_is_json_string()
    test_wrapped_in_outer_dict()
    test_missing_key()
    test_invalid_json()
    test_empty_input()
    test_list_json()
    print("\nAll tests passed.")
