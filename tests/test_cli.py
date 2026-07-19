"""Tests for the Typer CLI."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from rvrb_verify import verify
from rvrb_verify.cli import app

runner = CliRunner()


class TestCLIBasic:
    def test_help_succeeds(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Verify" in result.output

    def test_no_args_fails(self) -> None:
        result = runner.invoke(app, [])
        assert result.exit_code != 0
        assert "Error" in result.output or "empty" in result.output.lower()

    def test_stdin_input(self) -> None:
        result = runner.invoke(app, [], input="The sky is blue")
        # Will fail because no real provider, but CLI should parse args
        assert result.exit_code == 1
        assert "Unknown strategy" not in result.output


class TestCLIStrategy:
    def test_invalid_strategy_errors(self) -> None:
        result = runner.invoke(
            app,
            ["test claim", "--strategy", "nonexistent"],
        )
        assert result.exit_code == 1
        assert "Unknown strategy" in result.output

    def test_json_flag_accepted(self) -> None:
        result = runner.invoke(
            app,
            ["test claim", "--strategy", "fact-check", "--json"],
        )
        # Will fail because no real provider is available, but should
        # still parse CLI args correctly (error should be about provider)
        assert result.exit_code == 1
        # Make sure it's a provider error, not an arg parsing error
        assert "Unknown strategy" not in result.output
        assert "Error" in result.output or "provider" in result.output.lower()


class TestNonExistentStrategy:
    def test_verify_function_errors_on_bad_strategy(self) -> None:
        with pytest.raises(ValueError) as exc:
            verify("test", strategy="does-not-exist")
        assert "Unknown strategy" in str(exc.value)
