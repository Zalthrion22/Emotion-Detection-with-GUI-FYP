"""Unit tests for src/text_preprocessing.py.

"""
from __future__ import annotations

from src.text_preprocessing import (
    GOEMOTIONS_LABELS,
    _first_target_label,
    clean_text,
)


# clean_text
def test_clean_text_lowercases_and_strips_urls():
    out = clean_text("Visit https://example.com NOW!")
    assert "http" not in out
    assert "now!" in out


def test_clean_text_removes_mentions_and_placeholders():
    out = clean_text("Hey @bob, [NAME] said hi")
    assert "@bob" not in out
    assert "[name]" not in out
    assert "hi" in out


def test_clean_text_collapses_whitespace():
    assert clean_text("a  \t b\n c") == "a b c"


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_clean_text_strips_non_ascii():
    out = clean_text("hello ☃ world")  # snowman char
    assert "☃" not in out
    assert "hello" in out and "world" in out


# _first_target_label
def test_first_target_label_returns_mapped_class():
    # 17 = joy in GoEmotions index; maps to "joy"
    assert GOEMOTIONS_LABELS[17] == "joy"
    assert _first_target_label("17") == "joy"


def test_first_target_label_collapses_synonyms():
    # 0 = admiration -> "joy" via the synonym map in config.py
    assert GOEMOTIONS_LABELS[0] == "admiration"
    assert _first_target_label("0") == "joy"


def test_first_target_label_skips_unmapped_then_picks_next():
    # 11 = disgust (dropped); 25 = sadness (kept)
    assert GOEMOTIONS_LABELS[11] == "disgust"
    assert GOEMOTIONS_LABELS[25] == "sadness"
    assert _first_target_label("11,25") == "sadness"


def test_first_target_label_returns_none_when_no_mappable():
    # 8 = desire -- intentionally not in the target mapping
    assert GOEMOTIONS_LABELS[8] == "desire"
    assert _first_target_label("8") is None


def test_first_target_label_handles_garbage():
    assert _first_target_label("") is None
    assert _first_target_label("not_a_number") is None
    assert _first_target_label("999") is None  # out of range
