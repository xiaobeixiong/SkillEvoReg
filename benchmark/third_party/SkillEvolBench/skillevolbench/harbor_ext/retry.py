"""Selective Harbor retry policy for transient infrastructure failures."""

from __future__ import annotations


# Harbor serializes exception classes by name in ``ExceptionInfo``.  Keep the
# allow-list narrow: semantic failures and the 600-second AgentTimeoutError are
# deliberately absent.
RETRYABLE_INFRASTRUCTURE_EXCEPTIONS = frozenset({
    "InvalidEncryptedContentError",
    "ApiRateLimitError",
    "ApiInternalServerError",
    "ApiOverloadedError",
    "ApiConnectionClosedError",
    "ApiResponseStalledError",
    "UnknownApiError",
    "NetworkConnectionError",
})


def is_retryable_infrastructure_exception(exception_type: str | None) -> bool:
    return bool(exception_type) and exception_type in RETRYABLE_INFRASTRUCTURE_EXCEPTIONS


__all__ = [
    "RETRYABLE_INFRASTRUCTURE_EXCEPTIONS",
    "is_retryable_infrastructure_exception",
]
