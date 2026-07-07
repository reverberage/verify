"""Tests for rvrb_verify models."""

from __future__ import annotations

import pickle

import pytest
from pydantic import ValidationError

from rvrb_verify.models import (
    Claim,
    Evidence,
    ProviderError,
    QuotaExhaustedError,
    Source,
    ToolCall,
    ToolResult,
    Verdict,
    VerdictEnum,
)


class TestVerdictEnum:
    def test_values(self) -> None:
        assert VerdictEnum.TRUE == "true"
        assert VerdictEnum.FALSE == "false"
        assert VerdictEnum.UNCERTAIN == "uncertain"
        assert VerdictEnum.OPINION == "opinion"
        assert VerdictEnum.UNVERIFIABLE == "unverifiable"

    def test_membership(self) -> None:
        assert "true" in VerdictEnum._value2member_map_


class TestClaim:
    def test_basic(self) -> None:
        c = Claim(text="The sky is blue")
        assert c.text == "The sky is blue"
        assert c.domain == "fact-check"

    def test_custom_domain(self) -> None:
        c = Claim(text="Test", domain="legal")
        assert c.domain == "legal"

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            Claim(text="")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValidationError):
            Claim(text="   ")

    def test_json_roundtrip(self) -> None:
        c = Claim(text="Hello", domain="research")
        data = c.model_dump()
        restored = Claim.model_validate(data)
        assert restored.text == "Hello"
        assert restored.domain == "research"


class TestSource:
    def test_basic(self) -> None:
        s = Source(title="Test", url="https://example.com", snippet="content")
        assert s.title == "Test"
        assert s.url == "https://example.com"

    def test_default_retrieved_at(self) -> None:
        s = Source()
        assert s.retrieved_at is not None

    def test_json_roundtrip(self) -> None:
        s = Source(title="T", url="https://x.com", snippet="snip")
        data = s.model_dump(mode="json")
        restored = Source.model_validate(data)
        assert restored.title == "T"


class TestEvidence:
    def test_basic(self) -> None:
        ev = Evidence(reasoning="It matches", supports=True)
        assert ev.supports is True
        assert ev.reasoning == "It matches"

    def test_with_sources(self) -> None:
        src = Source(title="S", url="https://s.com")
        ev = Evidence(sources=[src])
        assert len(ev.sources) == 1

    def test_default_supports(self) -> None:
        ev = Evidence()
        assert ev.supports is True


class TestVerdict:
    def test_basic(self) -> None:
        v = Verdict(
            claim="The sky is blue",
            verdict=VerdictEnum.TRUE,
            confidence=0.95,
        )
        assert v.claim == "The sky is blue"
        assert v.verdict == VerdictEnum.TRUE

    def test_confidence_bounds(self) -> None:
        Verdict(claim="x", verdict=VerdictEnum.TRUE, confidence=0.0)
        Verdict(claim="x", verdict=VerdictEnum.TRUE, confidence=1.0)

    def test_confidence_zero(self) -> None:
        v = Verdict(claim="x", verdict=VerdictEnum.FALSE, confidence=0.0)
        assert v.confidence == 0.0

    def test_confidence_one(self) -> None:
        v = Verdict(claim="x", verdict=VerdictEnum.TRUE, confidence=1.0)
        assert v.confidence == 1.0

    def test_negative_confidence_raises(self) -> None:
        with pytest.raises(ValidationError):
            Verdict(claim="x", verdict=VerdictEnum.TRUE, confidence=-0.1)

    def test_with_evidence(self) -> None:
        ev = Evidence(reasoning="test")
        v = Verdict(
            claim="x",
            verdict=VerdictEnum.UNCERTAIN,
            confidence=0.5,
            evidence=[ev],
        )
        assert len(v.evidence) == 1

    def test_with_model_used(self) -> None:
        v = Verdict(
            claim="x",
            verdict=VerdictEnum.TRUE,
            confidence=0.9,
            model_used="qwen3.7-plus",
        )
        assert v.model_used == "qwen3.7-plus"

    def test_json_roundtrip(self) -> None:
        v = Verdict(
            claim="Test",
            verdict=VerdictEnum.TRUE,
            confidence=0.8,
            summary="OK",
        )
        data = v.model_dump(mode="json")
        restored = Verdict.model_validate(data)
        assert restored.claim == "Test"
        assert restored.summary == "OK"


class TestToolCall:
    def test_basic(self) -> None:
        tc = ToolCall(id="call_1", name="get_weather", arguments={"city": "Tokyo"})
        assert tc.id == "call_1"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "Tokyo"}

    def test_default_arguments(self) -> None:
        tc = ToolCall(id="c1", name="fn")
        assert tc.arguments == {}

    def test_validates_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            ToolCall(id="c1")  # type: ignore[call-arg]


class TestToolResult:
    def test_no_tool_calls(self) -> None:
        tr = ToolResult(content="Hello")
        assert tr.content == "Hello"
        assert tr.tool_calls == []

    def test_with_tool_calls(self) -> None:
        tc = ToolCall(id="c1", name="fn", arguments={"x": 1})
        tr = ToolResult(content=None, tool_calls=[tc])
        assert tr.content is None
        assert len(tr.tool_calls) == 1


class TestProviderError:
    def test_basic(self) -> None:
        err = ProviderError("model-1", 429, "rate limited")
        assert err.model_id == "model-1"
        assert err.status_code == 429
        assert err.body == "rate limited"
        assert "model-1" in str(err)

    def test_default_body(self) -> None:
        err = ProviderError("m", 500)
        assert err.body is None

    def test_pickling(self) -> None:
        err = pickle.loads(pickle.dumps(ProviderError("m", 401, "bad auth")))
        assert err.model_id == "m"
        assert err.status_code == 401


class TestQuotaExhaustedError:
    def test_is_provider_error(self) -> None:
        err = QuotaExhaustedError("m", 429, "quota")
        assert isinstance(err, ProviderError)

    def test_pickling(self) -> None:
        err = pickle.loads(pickle.dumps(QuotaExhaustedError("m", 429, "quota")))
        assert isinstance(err, QuotaExhaustedError)
