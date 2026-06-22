"""
converters/api_xml.py — REST XML / SOAP API to intermediate document converter.

Fetches XML responses (REST or SOAP), parses them with lxml, navigates
the document with XPath expressions, and maps each matched element to an
:class:`~converters.base.IntermediatePage`.  Supports two pagination
strategies (``next_url`` and ``offset_limit``) and optional SOAP
envelope wrapping.

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

_DEFAULT_SOAP_ENVELOPE_TEMPLATE = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">'
    "<soap:Body>{body}</soap:Body>"
    "</soap:Envelope>"
)


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class XmlPaginationConfig(BaseModel):
    """Pagination strategy for an XML API.

    Parameters
    ----------
    type:
        Pagination strategy: ``"next_url"`` follows a URL embedded in the
        response; ``"offset_limit"`` increments an offset query parameter.
    next_url_xpath:
        XPath expression pointing to the next-page URL text.
        Only used when ``type == "next_url"``.
    offset_param:
        Name of the offset query parameter.
        Only used when ``type == "offset_limit"``.
    limit_param:
        Name of the limit query parameter.
        Only used when ``type == "offset_limit"``.
    limit:
        Page size for offset/limit pagination.
    total_xpath:
        XPath expression pointing to the total item count text.
        Only used when ``type == "offset_limit"``.
    max_pages:
        Hard cap on the number of requests made to prevent infinite loops.
    """

    type: Literal["next_url", "offset_limit"]
    next_url_xpath: str = "//next/text()"
    offset_param: str = "offset"
    limit_param: str = "limit"
    limit: int = 100
    total_xpath: str = "//total/text()"
    max_pages: int = 1000


class ApiXmlMappingConfig(BaseModel):
    """Mapping configuration for an XML REST or SOAP API.

    Parameters
    ----------
    document_name:
        Human-readable name used in log messages.
    items_xpath:
        XPath expression selecting the elements to convert to pages
        (e.g. ``"//record"``).
    page_fields:
        Mapping of field roles to XPath expressions evaluated on each
        element.  Recognised keys: ``"title"``, ``"content"``,
        ``"metadata"`` (value may be a list of XPath strings).
    namespaces:
        XML namespace prefix-to-URI mapping passed to all XPath calls.
    pagination:
        Optional pagination configuration.
    soap_action:
        Value of the ``SOAPAction`` HTTP header.  When set, requests are
        sent as SOAP POSTs.
    soap_envelope_template:
        XML template for the SOAP envelope.  Must contain a ``{body}``
        placeholder.  When ``None`` and ``soap_action`` is set, the
        default SOAP 1.1 envelope is used.
    """

    document_name: str
    items_xpath: str
    page_fields: dict[str, Any]
    namespaces: dict[str, str] = {}
    pagination: XmlPaginationConfig | None = None
    soap_action: str | None = None
    soap_envelope_template: str | None = None


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


class ApiXmlConverter:
    """Convert a REST XML or SOAP API response to the intermediate format.

    This converter does *not* inherit from
    :class:`~converters.base.BaseConverter` because its source is a URL,
    not a file path.  Use :meth:`convert_from_url` as the main entry
    point.

    Parameters
    ----------
    config:
        Mapping configuration describing how to navigate the API response.
    headers:
        Optional HTTP headers merged into every request (auth tokens,
        custom headers, etc.).  SOAP-related headers are added
        automatically when required.
    """

    def __init__(
        self,
        config: ApiXmlMappingConfig,
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
    ) -> ApiXmlConverter:
        """Load configuration from a YAML or JSON file.

        Parameters
        ----------
        config_path:
            Path to a ``.yaml``, ``.yml``, or ``.json`` config file.
        headers:
            Optional HTTP headers forwarded to the constructor.

        Returns
        -------
        ApiXmlConverter

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
            config = ApiXmlMappingConfig.model_validate(data)
        except Exception as exc:
            raise ConversionError(
                f"Invalid config in {config_path}: {exc}"
            ) from exc

        return cls(config=config, headers=headers)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def convert_from_url(
        self,
        url: str,
        soap_body: str | None = None,
    ) -> IntermediateDocument:
        """Fetch the API and convert the response to the intermediate format.

        Parameters
        ----------
        url:
            URL of the API endpoint.
        soap_body:
            SOAP body content (without the envelope).  If provided and
            ``config.soap_envelope_template`` is not ``None`` (or
            ``config.soap_action`` is set), the body is wrapped in the
            SOAP envelope template before sending.

        Returns
        -------
        IntermediateDocument
            One :class:`~converters.base.IntermediatePage` per XML element
            matched by ``items_xpath``.

        Raises
        ------
        ConversionError
            If the HTTP request fails or the XML cannot be parsed.
        """
        logger.info(
            "Fetching XML API '%s' from %s", self._config.document_name, url
        )
        elements = self._fetch_all_items(url, soap_body)
        logger.info(
            "Fetched %d element(s) from '%s'",
            len(elements),
            self._config.document_name,
        )
        pages = [
            self._element_to_page(element, idx)
            for idx, element in enumerate(elements)
        ]
        return IntermediateDocument(pages=pages)

    # ------------------------------------------------------------------
    # Private: fetching
    # ------------------------------------------------------------------

    def _fetch_xml(self, url: str, soap_body: str | None = None) -> bytes:
        """Perform the HTTP request and return raw XML bytes.

        Uses POST with ``Content-Type: text/xml`` for SOAP requests
        (when ``soap_action`` or ``soap_envelope_template`` is set) and
        GET with ``Accept: application/xml`` otherwise.

        Parameters
        ----------
        url:
            URL to request.
        soap_body:
            Optional SOAP body content.

        Returns
        -------
        bytes
            Raw XML response body.

        Raises
        ------
        ConversionError
            On network error or non-2xx HTTP status.
        """
        try:
            import httpx
        except ImportError as exc:
            raise ConversionError(
                "httpx is required for XML API conversion. "
                "Install it with: uv add httpx"
            ) from exc

        is_soap = bool(
            self._config.soap_action or self._config.soap_envelope_template
        )

        if is_soap:
            template = (
                self._config.soap_envelope_template
                or _DEFAULT_SOAP_ENVELOPE_TEMPLATE
            )
            body_content = soap_body or ""
            request_body = template.format(body=body_content)

            request_headers = {
                **self._headers,
                "Content-Type": "text/xml; charset=utf-8",
            }
            if self._config.soap_action:
                request_headers["SOAPAction"] = f'"{self._config.soap_action}"'

            try:
                response = httpx.post(
                    url,
                    content=request_body.encode("utf-8"),
                    headers=request_headers,
                    timeout=httpx.Timeout(30.0),
                )
            except httpx.RequestError as exc:
                raise ConversionError(
                    f"Network error POSTing to {url}: {exc}"
                ) from exc
        else:
            request_headers = {
                **self._headers,
                "Accept": "application/xml",
            }
            try:
                response = httpx.get(
                    url,
                    headers=request_headers,
                    timeout=httpx.Timeout(30.0),
                )
            except httpx.RequestError as exc:
                raise ConversionError(
                    f"Network error fetching {url}: {exc}"
                ) from exc

        if response.is_error:
            raise ConversionError(
                f"HTTP {response.status_code} from {url}: {response.text}"
            )

        return response.content

    def _parse_xml(self, raw: bytes):  # type: ignore[return]
        """Parse raw XML bytes with lxml.

        Parameters
        ----------
        raw:
            Raw XML bytes.

        Returns
        -------
        lxml.etree._Element
            Root element of the parsed document.

        Raises
        ------
        ConversionError
            If ``lxml`` is not installed or the XML is malformed.
        """
        try:
            from lxml import etree  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ConversionError(
                "lxml is required for XML API conversion. "
                "Install it with: uv add lxml"
            ) from exc

        try:
            return etree.fromstring(raw)
        except etree.XMLSyntaxError as exc:
            raise ConversionError(f"XML parse error: {exc}") from exc

    def _fetch_all_items(
        self, url: str, soap_body: str | None = None
    ) -> list:
        """Fetch all XML elements, following pagination if configured.

        Parameters
        ----------
        url:
            Starting URL.
        soap_body:
            Optional SOAP body forwarded to :meth:`_fetch_xml`.

        Returns
        -------
        list
            Flat list of lxml elements across all pages.
        """
        pagination = self._config.pagination

        if pagination is None:
            raw = self._fetch_xml(url, soap_body)
            root = self._parse_xml(raw)
            return self._extract_elements(root)

        if pagination.type == "next_url":
            return self._fetch_next_url(url, soap_body, pagination)

        # offset_limit
        return self._fetch_offset_limit(url, soap_body, pagination)

    def _fetch_next_url(
        self,
        url: str,
        soap_body: str | None,
        pagination: XmlPaginationConfig,
    ) -> list:
        """Follow ``next`` URL links until exhausted.

        Parameters
        ----------
        url:
            Initial URL.
        soap_body:
            Optional SOAP body.
        pagination:
            Pagination configuration.

        Returns
        -------
        list
            lxml elements.
        """
        all_elements: list = []
        current_url: str | None = url
        pages_fetched = 0

        while current_url and pages_fetched < pagination.max_pages:
            raw = self._fetch_xml(current_url, soap_body)
            root = self._parse_xml(raw)
            all_elements.extend(self._extract_elements(root))
            pages_fetched += 1

            next_val = self._resolve_xpath(
                root, pagination.next_url_xpath, self._config.namespaces
            )
            current_url = next_val.strip() if next_val else None

        if pages_fetched >= pagination.max_pages:
            logger.warning(
                "Reached max_pages limit (%d) for '%s'",
                pagination.max_pages,
                self._config.document_name,
            )
        return all_elements

    def _fetch_offset_limit(
        self,
        url: str,
        soap_body: str | None,
        pagination: XmlPaginationConfig,
    ) -> list:
        """Paginate using offset and limit query parameters.

        Parameters
        ----------
        url:
            Base URL (without pagination parameters).
        soap_body:
            Optional SOAP body.
        pagination:
            Pagination configuration.

        Returns
        -------
        list
            lxml elements.
        """
        all_elements: list = []
        offset = 0
        pages_fetched = 0

        while pages_fetched < pagination.max_pages:
            paginated_url = (
                f"{url}"
                f"{'&' if '?' in url else '?'}"
                f"{pagination.offset_param}={offset}"
                f"&{pagination.limit_param}={pagination.limit}"
            )
            raw = self._fetch_xml(paginated_url, soap_body)
            root = self._parse_xml(raw)
            batch = self._extract_elements(root)
            all_elements.extend(batch)
            pages_fetched += 1

            if not batch:
                break

            total_str = self._resolve_xpath(
                root, pagination.total_xpath, self._config.namespaces
            )
            if total_str is not None:
                try:
                    total = int(total_str.strip())
                    offset += len(batch)
                    if offset >= total:
                        break
                except ValueError:
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
        return all_elements

    def _extract_elements(self, root) -> list:
        """Apply ``items_xpath`` to *root* and return matching elements.

        Parameters
        ----------
        root:
            lxml root element.

        Returns
        -------
        list
            Matching lxml elements.
        """
        try:
            results = root.xpath(
                self._config.items_xpath,
                namespaces=self._config.namespaces or None,
            )
        except Exception as exc:
            raise ConversionError(
                f"XPath error evaluating items_xpath "
                f"'{self._config.items_xpath}': {exc}"
            ) from exc

        if not isinstance(results, list):
            logger.warning(
                "items_xpath '%s' did not return a list for '%s'",
                self._config.items_xpath,
                self._config.document_name,
            )
            return []
        return results

    # ------------------------------------------------------------------
    # Private: mapping
    # ------------------------------------------------------------------

    def _element_to_page(self, element: Any, index: int) -> IntermediatePage:
        """Convert one lxml element to an :class:`IntermediatePage`.

        The ``page_fields`` config may contain:

        * ``"title"`` — XPath string → heading in the Markdown output.
        * ``"content"`` — XPath string → body text.
        * ``"metadata"`` — list of XPath strings → Markdown table.

        Parameters
        ----------
        element:
            An lxml element matched by ``items_xpath``.
        index:
            Zero-based element index used as the page index.

        Returns
        -------
        IntermediatePage
        """
        fields = self._config.page_fields
        ns = self._config.namespaces

        title_expr: str | None = fields.get("title")
        content_expr: str | None = fields.get("content")
        metadata_exprs: list[str] = fields.get("metadata") or []

        title = ""
        if title_expr:
            val = self._resolve_xpath(element, title_expr, ns)
            if val is None:
                logger.warning(
                    "title field '%s' resolved to None for element %d in '%s'",
                    title_expr,
                    index,
                    self._config.document_name,
                )
            else:
                title = val.strip()

        content = ""
        if content_expr:
            val = self._resolve_xpath(element, content_expr, ns)
            if val is None:
                logger.warning(
                    "content field '%s' resolved to None for element %d in '%s'",
                    content_expr,
                    index,
                    self._config.document_name,
                )
            else:
                content = val.strip()

        parts: list[str] = []
        if title:
            parts.append(f"## {title}")
        if content:
            parts.append(content)
        if metadata_exprs:
            table = self._metadata_to_markdown_table(element, metadata_exprs)
            if table:
                parts.append(table)

        markdown = "\n\n".join(parts)
        return IntermediatePage(index=index, markdown=markdown)

    def _resolve_xpath(
        self,
        element: Any,
        expr: str,
        namespaces: dict | None = None,
    ) -> str | None:
        """Evaluate an XPath expression and return the first result as a string.

        Parameters
        ----------
        element:
            lxml element to query.
        expr:
            XPath expression string.
        namespaces:
            Optional namespace prefix-to-URI mapping.

        Returns
        -------
        str | None
            String representation of the first match, or ``None`` if
            there are no matches.
        """
        try:
            results = element.xpath(expr, namespaces=namespaces or None)
        except Exception as exc:
            logger.warning("XPath error evaluating '%s': %s", expr, exc)
            return None

        if not results:
            return None

        first = results[0]
        if isinstance(first, str):
            return first
        if hasattr(first, "text") and first.text is not None:
            return first.text
        return str(first)

    def _metadata_to_markdown_table(
        self, element: Any, fields: list[str]
    ) -> str:
        """Build a two-column Markdown table from a list of XPath fields.

        The left column shows the last path segment as a label; the right
        column shows the resolved value.

        Parameters
        ----------
        element:
            lxml element to query.
        fields:
            List of XPath expressions to include as table rows.

        Returns
        -------
        str
            GFM Markdown table, or empty string if no fields resolve.
        """
        ns = self._config.namespaces
        rows: list[tuple[str, str]] = []

        for expr in fields:
            val = self._resolve_xpath(element, expr, ns)
            if val is None:
                continue
            # Derive a readable label from the last non-empty path segment
            label = (
                expr.rstrip(")")
                .rstrip("/")
                .split("/")[-1]
                .replace("text()", "")
                .replace("_", " ")
                .strip()
            )
            rows.append((label, val.strip().replace("|", "\\|")))

        if not rows:
            return ""

        lines: list[str] = [
            "| Campo | Valor |",
            "| --- | --- |",
        ]
        for label, value in rows:
            lines.append(f"| {label} | {value} |")
        return "\n".join(lines)
