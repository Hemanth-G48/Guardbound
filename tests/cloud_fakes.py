"""Offline OpenAI test harness: real SDK, fake HTTP transport.

The point of using ``httpx.MockTransport`` behind the *real* ``openai`` client —
rather than hand-rolled stub objects — is that the tests exercise the code paths
that actually ship: request serialisation, the SDK's own response parsing, and
the SDK's real exception classes. Nothing in this module makes a network call,
and no credential is required: the placeholder API key below never leaves the
process and is rejected by any real endpoint.

A response spec can be:
  * ``dict``            -> a 200 chat-completion payload (see ``completion``)
  * ``httpx.Response``  -> returned verbatim (status/error tests)
  * ``Exception``       -> raised from the transport (transport failure tests)
  * callable(request)   -> called for full control
"""
from __future__ import annotations

import json
from typing import Any

import httpx
import openai

# Obviously-not-a-credential: no "sk-" prefix, never sent to a real host.
TEST_API_KEY = "test-key-not-a-credential"
TEST_MODEL = "gpt-4o-mini"


def completion(
    content: str | None = "ok",
    *,
    refusal: str | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
    cached_tokens: int = 0,
    total_tokens: int | None = None,
    request_id: str = "chatcmpl-test",
    model: str = TEST_MODEL,
) -> dict[str, Any]:
    """A 200 chat-completions payload."""
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "refusal": refusal,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (
                total_tokens
                if total_tokens is not None
                else prompt_tokens + completion_tokens
            ),
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }


def error_response(status: int, message: str = "boom", **extra: Any) -> httpx.Response:
    return httpx.Response(
        status, json={"error": {"message": message, "type": "invalid_request_error", **extra}}
    )


class FakeOpenAIBackend:
    """A scripted OpenAI endpoint behind the real SDK client."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        self._specs: list[Any] = list(responses or [])
        self.requests: list[dict[str, Any]] = []
        self.transport = httpx.MockTransport(self._handle)
        self.client = openai.OpenAI(
            api_key=TEST_API_KEY,
            http_client=httpx.Client(transport=self.transport),
            max_retries=0,
        )

    def push(self, *specs: Any) -> "FakeOpenAIBackend":
        self._specs.extend(specs)
        return self

    @property
    def calls(self) -> int:
        return len(self.requests)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        try:
            body = json.loads(request.content or b"{}")
        except json.JSONDecodeError:
            body = {}
        self.requests.append(body)

        spec = self._specs.pop(0) if self._specs else completion()
        if callable(spec) and not isinstance(spec, httpx.Response):
            spec = spec(request)
        if isinstance(spec, Exception):
            raise spec
        if isinstance(spec, httpx.Response):
            return spec
        return httpx.Response(200, json=spec)
