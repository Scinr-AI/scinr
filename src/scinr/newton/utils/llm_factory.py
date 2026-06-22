"""
utils/llm_factory.py — Deprecated. Use scinr_config instead.

This module exists for backward compatibility only.
"""
from scinr.newton.config import get_llm, make_system_message  # noqa: F401

# Deprecated aliases
make_llm = get_llm
make_system_message_with_cache = make_system_message
