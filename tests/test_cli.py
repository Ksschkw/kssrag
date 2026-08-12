"""Tests for CLI helpers — system prompt loading in particular.

The silent-fallback bug: when `--system-prompt /some/missing/path.txt` was
passed, the CLI used the literal path string as the system prompt, so persona
instructions never reached the model (and there was no log line explaining why).
`load_system_prompt` now warns loudly when a path-like argument doesn't exist.
"""
import logging

from kssrag.cli import load_system_prompt


def test_no_prompt_returns_none():
    assert load_system_prompt(None) is None
    assert load_system_prompt("") is None


def test_reads_prompt_from_existing_file(tmp_path):
    f = tmp_path / "custom_prompt.txt"
    f.write_text("You are a helpful assistant.", encoding="utf-8")
    assert load_system_prompt(str(f)) == "You are a helpful assistant."


def test_literal_prompt_text_is_used_as_is():
    text = "You are Kosisochukwu Okafor."
    assert load_system_prompt(text) == text


def test_missing_path_like_prompt_warns_and_falls_back(tmp_path, caplog):
    missing = str(tmp_path / "custom_prompt.txt")  # does not exist
    with caplog.at_level(logging.WARNING, logger="KSSRAG"):
        result = load_system_prompt(missing)
    # Backward-compatible fallback (the raw argument) is preserved...
    assert result == missing
    # ...but the failure is now loud instead of silent.
    assert "not found" in caplog.text
    assert missing in caplog.text


def test_missing_absolute_path_prompt_warns(tmp_path, caplog):
    # Simulates a real mount path that doesn't exist (hermetic via tmp_path).
    missing = str(tmp_path / "run" / "secrets" / "custom_prompt.txt")
    with caplog.at_level(logging.WARNING, logger="KSSRAG"):
        result = load_system_prompt(missing)
    assert result == missing
    assert "not found" in caplog.text


def test_plain_text_without_path_shape_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="KSSRAG"):
        load_system_prompt("You are a helpful assistant.")
    assert "not found" not in caplog.text
