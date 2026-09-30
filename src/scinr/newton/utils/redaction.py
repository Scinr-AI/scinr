"""
utils/redaction.py — Scrub credentials out of text before it leaves the process.

Connection strings carry their credentials inline
(``mongodb://user:secret@host``), and both our own messages and some driver
errors echo them back. Anything that turns a connection string or an exception
into text that is logged or handed to a caller — a log line, a
``StageResult.errors`` entry, an exception message — should go through here.

Public API
----------
redact_uri(uri)
    Mask the password of one known connection string, keeping the user and host
    so the message is still useful for debugging.
redact_secrets(text)
    Scrub free text: mask the userinfo password of every ``scheme://user:pw@``
    it contains, plus any literal occurrence of the configured secrets
    (Neo4j password, MongoDB URI password, Mistral API key).
"""

from __future__ import annotations

import functools
import re
from urllib.parse import unquote_plus

MASK = "***"

# Literal secrets shorter than this are not scrubbed from free text: replacing
# a 4-letter password everywhere would mangle unrelated words. The structured
# ``scheme://user:pw@`` pattern is still masked whatever the password length.
_MIN_LITERAL_SECRET_LEN = 6

# ``scheme://userinfo@`` inside free text. The userinfo is greedy up to the LAST
# "@" before a delimiter, so an unescaped "@" in the password is still masked.
_URI_USERINFO_RE = re.compile(
    r"(?P<scheme>\b[A-Za-z][A-Za-z0-9+.\-]*://)(?P<userinfo>[^\s/?#'\"<>]*)@"
)


def _mask_userinfo(userinfo: str) -> str:
    user, sep, _password = userinfo.partition(":")
    return f"{user}:{MASK}" if sep else userinfo


def redact_uri(uri: str | None) -> str | None:
    """Return *uri* with the password in its userinfo replaced by ``***``.

    ``mongodb://app:s3cr3t@db1:27017,db2:27017/?authSource=admin`` becomes
    ``mongodb://app:***@db1:27017,db2:27017/?authSource=admin``.

    The split is on the last ``@`` of the whole string, so a password with
    unescaped reserved characters (``/``, ``?``, ``@``) is still fully masked.
    """
    if not uri or "://" not in uri:
        return uri
    scheme, _, rest = uri.partition("://")
    userinfo, sep, hosts = rest.rpartition("@")
    if not sep:
        return uri
    return f"{scheme}://{_mask_userinfo(userinfo)}@{hosts}"


def _uri_password(uri: str | None) -> str | None:
    if not uri or "://" not in uri:
        return None
    userinfo, sep, _ = uri.partition("://")[2].rpartition("@")
    if not sep:
        return None
    return userinfo.partition(":")[2] or None


@functools.lru_cache(maxsize=8)
def _literal_secrets(
    neo4j_password: str | None, mongodb_uri: str | None, mistral_api_key: str | None
) -> tuple[str, ...]:
    candidates: set[str] = set()
    mongo_pw = _uri_password(mongodb_uri)
    for value in (neo4j_password, mistral_api_key, mongo_pw):
        if value:
            candidates.add(value)
            candidates.add(unquote_plus(value))  # the decoded form drivers may echo
    # Longest first, so a secret containing another one is masked whole.
    return tuple(
        sorted((s for s in candidates if len(s) >= _MIN_LITERAL_SECRET_LEN), key=len, reverse=True)
    )


def _configured_secrets() -> tuple[str, ...]:
    try:
        from scinr.newton import config as _cfg_module  # deferred — config imports us

        cfg = _cfg_module._config
    except Exception:  # noqa: BLE001 — redaction must never raise
        return ()
    if cfg is None:
        return ()
    return _literal_secrets(cfg.neo4j_password or None, cfg.mongodb_uri, cfg.mistral_api_key)


def redact_secrets(text: str | None) -> str | None:
    """Return *text* with every credential it may contain replaced by ``***``.

    Masks the password of every ``scheme://user:password@`` URI in the text,
    then any literal occurrence (raw or URL-decoded) of the currently configured
    Neo4j password, MongoDB URI password and Mistral API key.
    """
    if not text:
        return text
    text = _URI_USERINFO_RE.sub(
        lambda m: f"{m.group('scheme')}{_mask_userinfo(m.group('userinfo'))}@", text
    )
    for secret in _configured_secrets():
        if secret in text:
            text = text.replace(secret, MASK)
    return text
