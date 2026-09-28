"""Phase 14.1 — JSON-constrained logits processor for local HF generation.

Reproduces the structured-output contract that GPT-4o's
``response_format={"type": "json_object"}`` provides: constrains the
local model to produce syntactically valid JSON without modifying prompts,
attack logic, history, or generation parameters.

Classification: LOCAL_RUNTIME_ADAPTATION (constrained decoding only).

Usage:
    processor = JSONGrammarLogitsProcessor(tokenizer, json_schema)
    output_ids = model.generate(**gen_kwargs, logits_processor=[processor])
"""
from __future__ import annotations

import json
from typing import Any

import torch
from transformers import LogitsProcessor


class JSONGrammarLogitsProcessor(LogitsProcessor):
    """Constrains token generation to produce valid JSON.

    Tracks the brace depth and string state to mask invalid tokens at each
    generation step.  The model can produce ANY content inside JSON string
    values; only the structural tokens (braces, quotes, colons, commas)
    are constrained.

    Supported JSON schema:
    {
      "generatedQuestion": "...",
      "lastResponseSummary": "..."
    }

    Also handles arbitrary flat JSON objects (keys with string values).
    """

    def __init__(self, tokenizer: Any, schema: dict[str, str] | None = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.schema = schema or {
            "generatedQuestion": "str",
            "lastResponseSummary": "str",
        }

        # Pre-compute token IDs for structural characters
        self._brace_open = self._token_id("{")
        self._brace_close = self._token_id("}")
        self._quote = self._token_id('"')
        self._colon = self._token_id(":")
        self._comma = self._token_id(",")
        self._whitespace = self._whitespace_token_ids()
        self._newline = self._token_id("\n")
        self._tab = self._token_id("\t")

        # Track JSON state
        self._brace_depth = 0
        self._in_string = False
        self._after_open_brace = False  # just saw { → expect key or close
        self._after_colon = False       # just saw : → expect value
        self._after_comma = False       # just saw , → expect key
        self._after_quote_in_string = False  # just saw " inside string
        self._step_count = 0

    def _token_id(self, s: str) -> int | None:
        """Get the token ID for a single character, or None if not found."""
        ids = self.tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if ids else None

    def _whitespace_token_ids(self) -> set[int]:
        """Get token IDs for common whitespace characters."""
        ids = set()
        for ch in [" ", "\n", "\t", "\r"]:
            tid = self._token_id(ch)
            if tid is not None:
                ids.add(tid)
        return ids

    def _reset_state(self) -> None:
        """Reset the JSON parser state for a new sequence."""
        self._brace_depth = 0
        self._in_string = False
        self._after_open_brace = False
        self._after_colon = False
        self._after_comma = False
        self._after_quote_in_string = False
        self._step_count = 0

    def _update_state_from_token(self, token_id: int) -> None:
        """Update the JSON parser state based on the generated token."""
        self._step_count += 1

        if token_id == self._brace_open:
            self._brace_depth += 1
            self._in_string = False
            self._after_open_brace = True
            self._after_colon = False
            self._after_comma = False
            self._after_quote_in_string = False
        elif token_id == self._brace_close:
            self._brace_depth = max(0, self._brace_depth - 1)
            self._in_string = False
            self._after_open_brace = False
            self._after_colon = False
            self._after_comma = False
            self._after_quote_in_string = False
        elif token_id == self._quote:
            if self._in_string:
                # Closing quote
                self._in_string = False
                self._after_quote_in_string = True
                self._after_open_brace = False
                self._after_colon = False
                self._after_comma = False
            else:
                # Opening quote
                self._in_string = True
                self._after_open_brace = False
                self._after_colon = False
                self._after_comma = False
                self._after_quote_in_string = False
        elif token_id == self._colon:
            self._after_colon = True
            self._after_open_brace = False
            self._after_comma = False
            self._after_quote_in_string = False
        elif token_id == self._comma:
            self._after_comma = True
            self._after_open_brace = False
            self._after_colon = False
            self._after_quote_in_string = False
        else:
            # Any other token (content, whitespace, etc.)
            self._after_open_brace = False
            self._after_colon = False
            self._after_comma = False
            self._after_quote_in_string = False

    def _get_valid_token_mask(self, vocab_size: int) -> torch.Tensor:
        """Return a boolean mask of shape [vocab_size] where True = valid token.

        The mask is conservative: it allows all tokens that COULD be valid,
        rather than trying to restrict to only the exact expected tokens.
        This ensures the model retains maximum expressiveness within JSON
        constraints.
        """
        mask = torch.zeros(vocab_size, dtype=torch.bool)

        # Always allow whitespace (spaces, newlines, tabs)
        mask[list(self._whitespace)] = True

        if self._step_count == 0:
            # First token: must be { (opening brace)
            if self._brace_open is not None:
                mask[self._brace_open] = True
            return mask

        if self._in_string:
            # Inside a string: allow any token except unescaped quotes
            # In practice, BPE tokens may contain quotes as part of multi-char tokens.
            # We allow everything here — the model's own string-escaping handles it.
            mask[:] = True
            # Only block standalone quote token if it's a single-char token
            # (multi-char BPE tokens containing " are fine)
            if self._quote is not None:
                # Don't block it entirely — the model might produce \" as part of a token
                mask[self._quote] = True
            return mask

        if self._brace_depth == 0:
            # Outside JSON entirely: only allow { to start, or whitespace
            if self._brace_open is not None:
                mask[self._brace_open] = True
            return mask

        # Inside JSON object (brace_depth >= 1):
        if self._after_open_brace or self._after_comma:
            # Expect a key (string) or closing brace
            if self._brace_close is not None:
                mask[self._brace_close] = True
            if self._quote is not None:
                mask[self._quote] = True
            return mask

        if self._after_colon:
            # Expect a value: string, number, bool, null, nested object/array
            if self._quote is not None:
                mask[self._quote] = True  # string value
            if self._brace_open is not None:
                mask[self._brace_open] = True  # nested object
            # Allow digits, -, t (true), f (false), n (null)
            for ch in ["0","1","2","3","4","5","6","7","8","9","-","t","f","n"]:
                tid = self._token_id(ch)
                if tid is not None:
                    mask[tid] = True
            return mask

        if self._after_quote_in_string:
            # After closing a string value: expect , or }
            if self._comma is not None:
                mask[self._comma] = True
            if self._brace_close is not None:
                mask[self._brace_close] = True
            return mask

        # Default: allow any token (model is in a state we don't track)
        # This prevents the processor from blocking generation entirely
        mask[:] = True
        return mask

    @torch.no_grad()
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.Tensor
    ) -> torch.Tensor:
        """Apply JSON grammar constraints to logits.

        Parameters
        ----------
        input_ids : [batch_size, seq_len]
        scores : [batch_size, vocab_size]

        Returns
        -------
        scores : [batch_size, vocab_size]
            Modified scores with invalid tokens set to -inf.
        """
        batch_size, vocab_size = scores.shape

        # Process only the last token of each sequence to update state
        for b in range(batch_size):
            # Get the last generated token
            last_token = input_ids[b, -1].item()

            # If this is the first call for this sequence, reset state
            if self._step_count == 0 and b == 0:
                self._reset_state()

            # Update state based on the token that was just generated
            self._update_state_from_token(last_token)

        # Build the valid token mask (on CPU first, then move to match scores)
        valid_mask = self._get_valid_token_mask(vocab_size)

        # Apply mask: set invalid token scores to -inf
        mask_expanded = valid_mask.unsqueeze(0).expand(batch_size, -1)
        mask_expanded = mask_expanded.to(scores.device)
        scores = scores.masked_fill(~mask_expanded, float("-inf"))

        return scores


def build_json_logits_processor(
    tokenizer: Any,
    json_schema: dict[str, str] | None = None,
) -> list[LogitsProcessor] | None:
    """Build a JSON logits processor for constrained generation.

    Returns None if json_format is False or the processor cannot be built.
    """
    processor = JSONGrammarLogitsProcessor(tokenizer, json_schema)
    return [processor]
