"""classify_error のテスト — claude_fallback.py のエラー分類コアロジック."""

from __future__ import annotations

import pytest

from claude_fallback import classify_error


@pytest.fixture
def cfg() -> dict:
    """fallback-config.json と同じ構造のテスト用設定."""
    return {
        "fallback": {
            "fallback_on": {
                "http_status_codes": [429, 500, 502, 503, 504],
                "error_keywords": [
                    "timeout",
                    "timed out",
                    "econnreset",
                    "connection reset",
                ],
            },
            "do_not_fallback_on": {
                "http_status_codes": [400, 401, 403, 404, 422],
                "error_keywords": [
                    "invalid api key",
                    "authentication",
                    "unauthorized",
                    "forbidden",
                    "bad request",
                ],
            },
        }
    }


class TestClassifyErrorDoNotFallback:
    """リトライすべきでない（non_retryable）エラーの分類."""

    def test_do_not_http_status_401(self, cfg: dict) -> None:
        assert classify_error("Request failed with 401", cfg) == "non_retryable_http"

    def test_do_not_http_status_403(self, cfg: dict) -> None:
        assert classify_error("got 403 forbidden", cfg) == "non_retryable_http"

    def test_do_not_keyword_unauthorized(self, cfg: dict) -> None:
        assert classify_error("user is unauthorized", cfg) == "non_retryable_keyword"

    def test_do_not_keyword_invalid_api_key(self, cfg: dict) -> None:
        assert classify_error("invalid api key supplied", cfg) == "non_retryable_keyword"


class TestClassifyErrorRetryable:
    """リトライ可能（retryable）エラーの分類."""

    def test_retryable_http_429(self, cfg: dict) -> None:
        assert classify_error("rate limited: 429", cfg) == "retryable_http"

    def test_retryable_http_503(self, cfg: dict) -> None:
        assert classify_error("server returned 503", cfg) == "retryable_http"

    def test_retryable_keyword_timeout(self, cfg: dict) -> None:
        assert classify_error("request timeout occurred", cfg) == "retryable_keyword"

    def test_retryable_keyword_connection_reset(self, cfg: dict) -> None:
        assert classify_error("connection reset by peer", cfg) == "retryable_keyword"


class TestClassifyErrorEdgeCases:
    """境界条件・優先順位・正規化."""

    def test_unknown_when_nothing_matches(self, cfg: dict) -> None:
        assert classify_error("some benign log message", cfg) == "unknown"

    def test_empty_text_returns_unknown(self, cfg: dict) -> None:
        assert classify_error("", cfg) == "unknown"

    def test_none_text_returns_unknown(self, cfg: dict) -> None:
        assert classify_error(None, cfg) == "unknown"  # type: ignore[arg-type]

    def test_case_insensitive(self, cfg: dict) -> None:
        """大文字の TIMEOUT が小文字 timeout にマッチ."""
        assert classify_error("REQUEST TIMEOUT", cfg) == "retryable_keyword"

    def test_do_not_takes_precedence_over_fallback(self, cfg: dict) -> None:
        """do_not と fallback_on が両方マッチする場合、do_not が優先."""
        # 401 (do_not) と timeout (fallback_on) が両方 → non_retryable_http
        text = "401 unauthorized but also timeout"
        assert classify_error(text, cfg) == "non_retryable_http"

    def test_http_status_word_boundary(self, cfg: dict) -> None:
        """4290 のような部分マッチを回避（単語境界）."""
        # "4290" は 429 にマッチしてはいけない
        assert classify_error("error code 4290 something", cfg) == "unknown"

    def test_do_not_keyword_precedence_over_fallback_keyword(self, cfg: dict) -> None:
        """do_not keyword が fallback keyword より優先."""
        text = "bad request caused timeout"
        assert classify_error(text, cfg) == "non_retryable_keyword"
