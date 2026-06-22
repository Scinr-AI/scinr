"""
converters/api_json.py — REST JSON API to intermediate document converter.

Fetches one or more pages of a JSON API, navigates the response with
JSONPath expressions, and maps each item to an
:class:`~converters.base.IntermediatePage`.  Supports two pagination
strategies: following a ``next`` URL field (``next_url``) and classic
offset/limit pagination (``offset_limit``).

Config files can be YAML or JSON and are validated with Pydantic v2.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from scinr.newton.converters.base import ConversionError, IntermediateDocument, IntermediatePage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class PaginationConfig(BaseModel):
    """Pagination strategy for a JSON API.

    Parameters
    ----------
    type:
        Pagination strategy: ``"next_url"`` follows a URL embedded in the
        response; ``"offset_limit"`` increments an offset query parameter.
    next_url_path:
        JSONPath expression pointing to the next-page URL in the response.
        Only used when ``type == "next_url"``.
    offset_param:
        Name of the offset query parameter.
        Only used when ``type == "offset_limit"``.
    limit_param:
        Name of the limit query parameter.
        Only used when ``type == "offset_limit"``.
    limit:
        Page size for offset/limit pagination.
    total_path:
        JSONPath expression pointing to the total item count in the
        response.  Only used when ``type == "offset_limit"``.
    max_pages:
        Hard cap on the number of requests made to prevent infinite loops.
    """

    type: Literal["next_url", "offset_limit"]
    next_url_path: str = "$.next"
    offset_param: str = "offset"
    limit_param: str = "limit"
    limit: int = 100
    total_path: str = "$.total"
    max_pages: int = 1000


class ApiMappingConfig(BaseModel):
    """Mapping configuration for a JSON REST API.

    Parameters
    ----------
    document_name:
        Human-readable name used in log messages.
    items_path:
        JSONPath expression pointing to the array of items in each
        API response.  Use ``"$"`` when the response is directly a list.
    page_fields:
        Mapping of field roles to JSONPath expressions evaluated on each
        item.  Recognised keys: ``"title"``, ``"content"``,
        ``"metadata"`` (value may be a list of JSONPath strings).
    pagination:
        Optional pagination configuration.
    """

    document_name: str
    items_path: str
    page_fields: dict[str, Any]
    pagination: PaginationConfig | None = None


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


class ApiJsonConverter:
    """Convert a REST JSON API response to the intermediate document format.

    This converter does *not* inherit from
    :class:`~converters.base.BaseConverter` because its source is a URL,
    not a file path.  Use :meth:`convert_from_url` as the main entry
    point.

    Parameters
    ----------
    config:
        Mapping configuration describing how to navigate the API response.
    headers:
        Optional HTTP headers sent with every request (e.g. auth tokens).
    """

    def __init__(
        self,
        config: ApiMappingConfig,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._config = config
        self._headers: dict[str, str] = headers or {}

    # ------------------------------------------------------------------
    # Class-method constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_config_file(
        cls,
        config_path: Path,
        headers: dict[str, str] | None = None,
    ) -> ApiJsonConverter:
        """Load configuration from a YAML or JSON file.

        Parameters
        ----------
        config_path:
            Path to a ``.yaml``, ``.yml``, or ``.json`` config file.
        headers:
            Optional HTTP headers forwarded to the constructor.

        Returns
        -------
        ApiJsonConverter

        Raises
        ------
        ConversionError
            If the file cannot be read, parsed, or validated.
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        suffix = config_path.suffix.lower()
        try:
            raw_text = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConversionError(
                f"Cannot read config file {config_path}: {exc}"
            ) from exc

        if suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ConversionError(
                    "PyYAML is required to load YAML config files. "
                    "Install it with: uv add pyyaml"
                ) from exc
            try:
                data = yaml.safe_load(raw_text)
            except yaml.YAMLError as exc:
                raise ConversionError(
                    f"YAML parse error in {config_path}: {exc}"
                ) from exc
        elif suffix == ".json":
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                raise ConversionError(
                    f"JSON parse error in {config_path}: {exc}"
                ) from exc
        else:
            raise ConversionError(
                f"Unsupported config file extension '{suffix}'. "
                "Use .yaml, .yml, or .json."
            )

        try:
            config = ApiMappingConfig.model_validate(data)
        except Exception as exc:
            raise ConversionError(
                f"Invalid config in {config_path}: {exc}"
            ) from exc

        return cls(config=config, headers=headers)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def convert_from_url(self, url: str) -> IntermediateDocument:
        """Fetch the API and convert the response to the intermediate format.

        Parameters
        ----------
        url:
            Base URL of the API endpoint.

        Returns
        -------
        IntermediateDocument
            One :class:`~converters.base.IntermediatePage` per item
            extracted from the API response.

        Raises
        ------
        ConversionError
            If the HTTP request fails or the response cannot be parsed.
        """
        logger.info(
            "Fetching JSON API '%s' from %s", self._config.document_name, url
        )
        items = self._fetch_all_items(url)
        logger.info(
            "Fetched %d item(s) from '%s'", len(items), self._config.document_name
        )
        pages = [self._item_to_page(item, idx) for idx, item in enumerate(items)]
        return IntermediateDocument(pages=pages)

    # ------------------------------------------------------------------
    # Private: fetching
    # ------------------------------------------------------------------

    def _fetch_all_items(self, url: str) -> list[dict]:
        """Fetch all items from the API, following pagination if configured.

        Parameters
        ----------
        url:
            Starting URL.

        Returns
        -------
        list[dict]
            Flat list of item dicts across all pages.
        """
        try:
            import httpx
        except ImportError as exc:
            raise ConversionError(
                "httpx is required for JSON API conversion. "
                "Install it with: uv add httpx"
            ) from exc

        pagination = self._config.pagination

        if pagination is None:
            data = self._get_json(httpx, url)
            return self._extract_items(data)

        if pagination.type == "next_url":
            return self._fetch_next_url(httpx, url, pagination)

        # offset_limit
        return self._fetch_offset_limit(httpx, url, pagination)

    def _fetch_next_url(
        self,
        httpx: Any,
        url: str,
        pagination: PaginationConfig,
    ) -> list[dict]:
        """Follow ``next`` URL links until exhausted.

        Parameters
        ----------
        httpx:
            The imported httpx module.
        url:
            Initial URL.
        pagination:
            Pagination configuration.

        Returns
        -------
        list[dict]
        """
        all_items: list[dict] = []
        current_url: str | None = url
        pages_fetched = 0

        while current_url and pages_fetched < pagination.max_pages:
            data = self._get_json(httpx, current_url)
            all_items.extend(self._extract_items(data))
            pages_fetched += 1

            next_val = self._resolve_jsonpath(data, pagination.next_url_path)
            current_url = next_val if isinstance(next_val, str) else None

        if pages_fetched >= pagination.max_pages:
            logger.warning(
                "Reached max_pages limit (%d) for '%s'",
                pagination.max_pages,
                self._config.document_name,
            )
        return all_items

    def _fetch_offset_limit(
        self,
        httpx: Any,
        url: str,
        pagination: PaginationConfig,
    ) -> list[dict]:
        """Paginate using offset and limit query parameters.

        Parameters
        ----------
        httpx:
            The imported httpx module.
        url:
            Base URL (without pagination parameters).
        pagination:
            Pagination configuration.

        Returns
        -------
        list[dict]
        """
        all_items: list[dict] = []
        offset = 0
        pages_fetched = 0

        while pages_fetched < pagination.max_pages:
            paginated_url = (
                f"{url}"
                f"{'&' if '?' in url else '?'}"
                f"{pagination.offset_param}={offset}"
                f"&{pagination.limit_param}={pagination.limit}"
            )
            data = self._get_json(httpx, paginated_url)
            batch = self._extract_items(data)
            all_items.extend(batch)
            pages_fetched += 1

            if not batch:
                break

            total_val = self._resolve_jsonpath(data, pagination.total_path)
            if total_val is not None:
                try:
                    total = int(total_val)
                    offset += len(batch)
                    if offset >= total:
                        break
                except (TypeError, ValueError):
                    offset += len(batch)
            else:
                offset += len(batch)

            if len(batch) < pagination.limit:
                break

        if pages_fetched >= pagination.max_pages:
            logger.warning(
                "Reached max_pages limit (%d) for '%s'",
                pagination.max_pages,
                self._config.document_name,
            )
        return all_items

    def _get_json(self, httpx: Any, url: str) -> Any:
        """Perform a GET request and return the parsed JSON.

        Parameters
        ----------
        httpx:
            The imported httpx module.
        url:
            URL to fetch.

        Returns
        -------
        Any
            Parsed JSON (typically a dict or list).

        Raises
        ------
        ConversionError
            On network error, non-2xx status, or JSON parse failure.
        """
        try:
            response = httpx.get(
                url,
                headers=self._headers,
                timeout=httpx.Timeout(30.0),
            )
        except httpx.RequestError as exc:
            raise ConversionError(
                f"Network error fetching {url}: {exc}"
            ) from exc

        if response.is_error:
            raise ConversionError(
                f"HTTP {response.status_code} fetching {url}: {response.text}"
            )

        try:
            return response.json()
        except Exception as exc:
            raise ConversionError(
                f"Cannot parse response from {url} as JSON: {exc}"
            ) from exc

    def _extract_items(self, data: Any) -> list[dict]:
        """Extract the items array from an API response.

        Parameters
        ----------
        data:
            Full parsed JSON response.

        Returns
        -------
        list[dict]
            Extracted items.  If the resolved value is a dict (not a
            list), it is wrapped in a single-element list with a warning.
        """
        result = self._resolve_jsonpath(data, self._config.items_path)

        if result is None:
            logger.warning(
                "items_path '%s' resolved to None for document '%s'",
                self._config.items_path,
                self._config.document_name,
            )
            return []

        if isinstance(result, dict):
            logger.warning(
                "items_path '%s' resolved to a dict (expected list) for '%s'; "
                "wrapping in list",
                self._config.items_path,
                self._config.document_name,
            )
            return [result]

        if isinstance(result, list):
            return result  # type: ignore[return-value]

        logger.warning(
            "items_path '%s' resolved to unexpected type %s for '%s'",
            self._config.items_path,
            type(result).__name__,
            self._config.document_name,
        )
        return []

    # ------------------------------------------------------------------
    # Private: mapping
    # ------------------------------------------------------------------

    def _item_to_page(self, item: dict, index: int) -> IntermediatePage:
        """Convert one API item dict to an :class:`IntermediatePage`.

        The ``page_fields`` config may contain:

        * ``"title"`` — JSONPath string → heading in the Markdown output.
        * ``"content"`` — JSONPath string → body text.
        * ``"metadata"`` — list of JSONPath strings → Markdown table.

        Parameters
        ----------
        item:
            A single item dict from the API response.
        index:
            Zero-based item index used as the page index.

        Returns
        -------
        IntermediatePage
        """
        fields = self._config.page_fields

        title_expr: str | None = fields.get("title")
        content_expr: str | None = fields.get("content")
        metadata_exprs: list[str] = fields.get("metadata") or []

        title = ""
        if title_expr:
            val = self._resolve_jsonpath(item, title_expr)
            if val is None:
                logger.warning(
                    "title field '%s' resolved to None for item %d in '%s'",
                    title_expr,
                    index,
                    self._config.document_name,
                )
            else:
                title = str(val)

        content = ""
        if content_expr:
            val = self._resolve_jsonpath(item, content_expr)
            if val is None:
                logger.warning(
                    "content field '%s' resolved to None for item %d in '%s'",
                    content_expr,
                    index,
                    self._config.document_name,
                )
            else:
                content = str(val)

        parts: list[str] = []
        if title:
            parts.append(f"## {title}")
        if content:
            parts.append(content)
        if metadata_exprs:
            table = self._metadata_to_markdown_table(item, metadata_exprs)
            if table:
                parts.append(table)

        markdown = "\n\n".join(parts)
        return IntermediatePage(index=index, markdown=markdown)

    def _resolve_jsonpath(self, data: Any, expr: str) -> Any:
        """Evaluate a JSONPath expression against *data*.

        Parameters
        ----------
        data:
            The data to query.
        expr:
            JSONPath expression string (e.g. ``"$.items[*].name"``).

        Returns
        -------
        Any
            The value of the first match, or ``None`` if there are no
            matches.

        Raises
        ------
        ConversionError
            If ``jsonpath_ng`` is not installed.
        """
        try:
            from jsonpath_ng.ext import parse as jp_parse  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ConversionError(
                "jsonpath-ng is required for JSON API conversion. "
                "Install it with: uv add jsonpath-ng"
            ) from exc

        try:
            matches = jp_parse(expr).find(data)
        except Exception as exc:
            logger.warning("JSONPath error evaluating '%s': %s", expr, exc)
            return None

        if not matches:
            return None
        return matches[0].value

    def _metadata_to_markdown_table(
        self, item: dict, fields: list[str]
    ) -> str:
        """Build a two-column Markdown table from a list of JSONPath fields.

        The left column shows the last path segment as a label; the right
        column shows the resolved value.

        Parameters
        ----------
        item:
            The API item dict to query.
        fields:
            List of JSONPath expressions to include as table rows.

        Returns
        -------
        str
            GFM Markdown table, or empty string if no fields resolve.
        """
        rows: list[tuple[str, str]] = []
        for expr in fields:
            val = self._resolve_jsonpath(item, expr)
            if val is None:
                continue
            # Derive a readable label from the last path segment
            label = expr.rstrip(")").split(".")[-1].replace("_", " ").strip()
            rows.append((label, str(val).replace("|", "\\|")))

        if not rows:
            return ""

        lines: list[str] = [
            "| Campo | Valor |",
            "| --- | --- |",
        ]
        for label, value in rows:
            lines.append(f"| {label} | {value} |")
        return "\n".join(lines)
