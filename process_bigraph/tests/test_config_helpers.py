"""Tests for process_bigraph.config_helpers.normalize_config_list (moved from
viva-superpowers; v2ecoli #3)."""
import pytest

from process_bigraph.config_helpers import normalize_config_list


def test_list_passthrough():
    assert normalize_config_list([0.2, 0.5]) == [0.2, 0.5]


def test_tuple_becomes_list():
    assert normalize_config_list((0.2, 0.5)) == [0.2, 0.5]


def test_int_keyed_dict_ordered_by_index():
    """bigraph-schema rewrap: {0: low, 1: high}."""
    assert normalize_config_list({0: 0.2, 1: 0.5}) == [0.2, 0.5]
    # Out-of-order keys still sort by index.
    assert normalize_config_list({1: 0.5, 0: 0.2}) == [0.2, 0.5]


def test_string_keyed_dict_ordered_by_index():
    """JSON round-trip stringifies the int keys: {"0": low, "1": high}."""
    assert normalize_config_list({"0": 0.2, "1": 0.5}) == [0.2, 0.5]
    assert normalize_config_list({"1": 0.5, "0": 0.2}) == [0.2, 0.5]


def test_explicit_key_form_default_low_high():
    assert normalize_config_list({"low": 0.2, "high": 0.5}) == [0.2, 0.5]
    # Order follows key_names, not dict insertion order.
    assert normalize_config_list({"high": 0.5, "low": 0.2}) == [0.2, 0.5]


def test_explicit_key_form_custom_names():
    assert normalize_config_list(
        {"min": 1, "max": 9}, key_names=("min", "max")
    ) == [1, 9]
    assert normalize_config_list(
        {"x": 1, "y": 2, "z": 3}, key_names=("x", "y", "z")
    ) == [1, 2, 3]


def test_scalar_wrapped_as_single_element():
    assert normalize_config_list(0.3) == [0.3]


def test_none_is_empty_list():
    assert normalize_config_list(None) == []


def test_length_assertion_passes():
    assert normalize_config_list([0.2, 0.5], length=2) == [0.2, 0.5]


def test_length_assertion_fails():
    with pytest.raises(ValueError, match="expected 2 element"):
        normalize_config_list([0.2, 0.5, 0.9], length=2)


def test_unrecognized_dict_keys_raise():
    with pytest.raises(ValueError, match="neither one of"):
        normalize_config_list({"lo": 0.2, "hi": 0.5})


def test_bool_keys_are_not_treated_as_index():
    """`True`/`False` are int subclasses; they must not be read as 1/0."""
    with pytest.raises(ValueError):
        normalize_config_list({True: 0.5, False: 0.2})


def test_three_element_numeric_dict():
    assert normalize_config_list({0: "a", 1: "b", 2: "c"}) == ["a", "b", "c"]
