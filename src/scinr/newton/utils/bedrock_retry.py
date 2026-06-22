"""
utils/bedrock_retry.py — Deprecated. Use utils/llm_retry instead.

This module exists for backward compatibility only. It re-exports
with_bedrock_retry from utils.llm_retry.
"""
from scinr.newton.utils.llm_retry import with_bedrock_retry, with_llm_retry  # noqa: F401
