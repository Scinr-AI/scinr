"""
converters/main.py — Conversion API of the converters module.

Converts source files and API responses to the intermediate JSON format
consumed by the extraction stage. ``run_preprocess()`` and ``run_pipeline()``
call these functions; they can also be called directly.

Usage examples
--------------
Convert every file of a folder::

    written, failures = await convert_folder(Path("files/"), Path("data/json/"))

Convert a single file::

    output_path = await convert_single_file(Path("files/report.pdf"), Path("data/json/"))

Fetch from a JSON or XML/SOAP API::

    output_path = convert_api(
        config_path=Path("files/api_config.yaml"),
        url="https://api.example.com/records",
        output_dir=Path("data/json/"),
        headers={"Authorization": "Bearer token"},
        api_type="json",  # or "xml"
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scinr.newton.storage.base import PageRepository, RawFileRepository

from scinr.newton.converters.api_json import ApiJsonConverter
from scinr.newton.converters.api_xml import ApiXmlConverter
from scinr.newton.converters.base import (
    BaseConverter,
    ConversionError,
    IntermediateDocument,
    UnsupportedFormatError,
)
from scinr.newton.converters.registry import get_converter
from scinr.newton.utils.redaction import redact_uri

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _guess_content_type(path: Path) -> str:
    """Infer MIME type from file extension. Falls back to 'application/octet-stream'."""
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


async def _run_convert(converter: BaseConverter, source: Path) -> IntermediateDocument:
    """Run converter.convert(source) respecting its sync/async contract.

    Async converters (converter.is_async is True, e.g. PdfConverter) are
    awaited directly on the event loop — their blocking work is genuine
    network I/O already expressed as native coroutines, so no thread is
    needed. Sync converters are dispatched to asyncio.to_thread() so their
    CPU/local-disk-bound work does not block the event loop, allowing
    other tasks (bounded by the caller's semaphore) to make real progress
    concurrently.
    """
    if getattr(converter, "is_async", False):
        return await converter.convert(source)
    return await asyncio.to_thread(converter.convert, source)


def _load_document_name(config_path: Path) -> str:
    """Read the ``document_name`` field from a YAML or JSON config file.

    Parameters
    ----------
    config_path:
        Path to a ``.yaml``, ``.yml``, or ``.json`` config file.

    Returns
    -------
    str
        Value of the ``document_name`` key.

    Raises
    ------
    KeyError
        If the ``document_name`` key is absent from the config.
    ValueError
        If the file extension is not supported.
    """
    suffix = config_path.suffix.lower()
    raw_text = config_path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(raw_text)
    elif suffix == ".json":
        data = json.loads(raw_text)
    else:
        raise ValueError(
            f"Unsupported config file extension '{suffix}'. Use .yaml, .yml, or .json."
        )

    if "document_name" not in data:
        raise KeyError(f"'document_name' key not found in config file: {config_path}")
    return str(data["document_name"])


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


async def convert_one(
    entry: Path,
    output_dir: Path | None,
    dry_run: bool = False,
    raw_file_repo: RawFileRepository | None = None,
    page_repo: PageRepository | None = None,
    _relative_prefix: Path | None = None,  # internal: relative path from the original input_dir
    context_instructions: str | None = None,
    parallel_docs: int = 1,
    *,
    tenant_id: str | None = None,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
) -> tuple[list[tuple[Path, Path | None, IntermediateDocument]], list[tuple[Path, str]]]:
    """Convert a single directory entry (a file or a subdirectory).

    This is the per-entry unit of work factored out of :func:`convert_folder`
    so that callers can schedule many entries concurrently instead of looping
    over them sequentially.

    Concurrency contract
    ---------------------
    ``convert_one`` does **not** create or acquire a semaphore for *this*
    entry — the caller (:func:`convert_folder`) is responsible for acquiring
    a concurrency-limiting semaphore before invoking ``convert_one`` and
    holding it for the entry's entire duration (including, when *entry* is a
    subdirectory, the whole nested conversion).

    Design decision — per-level (not global) concurrency budget
    -------------------------------------------------------------
    When *entry* is a subdirectory, this delegates to :func:`convert_folder`,
    passing *parallel_docs* through but **not** any semaphore instance. Each
    recursive :func:`convert_folder` call creates its own fresh
    ``asyncio.Semaphore(parallel_docs)`` for its own children rather than
    reusing/sharing the parent level's semaphore instance.

    This is a deliberate choice, not an oversight: sharing a single semaphore
    instance across recursion depths while the parent entry (the directory
    itself) holds it for the entire nested conversion would self-deadlock —
    the nested level would try to acquire the very same semaphore slot that
    its own parent is holding, and with the default ``parallel_docs=1`` this
    deadlocks on *every* tree with subdirectories (verified empirically while
    implementing this). Bounding concurrency independently at each directory
    level avoids this entirely, is trivially correct, and still gives the
    caller a meaningful, predictable cap ("no more than `parallel_docs`
    entries convert concurrently within any single directory") at the cost of
    not being a strict *global* cap across a deeply/widely nested tree (worst
    case, total in-flight conversions can be higher than `parallel_docs` if
    multiple sibling directories are each fanning out their own children
    concurrently). For `parallel_docs=1` — the default, and the only value
    required to be behaviourally identical to the old sequential code — this
    distinction is moot: every level is sequential, so the whole tree is
    processed one entry at a time, exactly as before.

    In addition to this per-level semaphore separation, the actual
    conversion work for each entry (``converter.convert(entry)``) is
    dispatched via ``_run_convert()`` rather than called directly: sync
    converters run in a worker thread (``asyncio.to_thread()``), while
    async converters (e.g. ``PdfConverter``, whose conversion is genuine
    network I/O) are awaited directly on the event loop. This matters
    because before this dispatch existed, a blocking in-coroutine call to
    ``converter.convert()`` monopolised the event loop for its entire
    duration, silently negating whatever concurrency the semaphore had
    scheduled — i.e. `parallel_docs > 1` had no real effect. Dispatching
    through ``_run_convert()`` is what makes the semaphore's scheduled
    parallelism translate into actual concurrent progress.

    Parameters
    ----------
    entry:
        The file or subdirectory to convert.
    output_dir:
        Directory where converted JSON files will be written, or ``None`` to
        **not write anything to disk**: the converted
        :class:`~converters.base.IntermediateDocument` is only returned in
        memory (the returned path is ``None``). The caller decides whether the
        JSON is persisted by passing (or not) a directory; ``None`` is only
        valid when *entry* is a file — a directory entry needs a real output
        directory to mirror its structure into.
    dry_run:
        If ``True``, logs what would be done but writes nothing.
    raw_file_repo:
        Optional :class:`~storage.base.RawFileRepository` for persisting the
        original binary file.  When ``None``, raw storage is skipped.
    page_repo:
        Optional :class:`~storage.base.PageRepository` for persisting
        converted pages.  When ``None``, page storage is skipped.
    _relative_prefix:
        Internal parameter for recursive calls. Tracks the relative path
        of *entry*'s parent directory from the original root. Do not pass
        this manually — with one documented exception:
        ``scinr.newton.pipeline._process_document_unit()`` (the per-document
        orchestration engine) legitimately passes it explicitly, since it
        calls ``convert_one()`` directly for a single ``raw_file`` unit
        without going through :func:`convert_folder`'s own recursive
        bookkeeping, and therefore has to supply the unit's precomputed
        relative directory itself. That call site is the one intentional,
        documented bypass of this "internal only" contract; no other caller
        should pass this argument.
    context_instructions:
        Optional free-text context injected into the ``IntermediateDocument``
        before it is written to disk.
    parallel_docs:
        Forwarded to the recursive :func:`convert_folder` call when *entry*
        is a subdirectory (see "Design decision" above). Ignored for file
        entries.
    tenant_id:
        Tenant owning the converted file(s) (``None`` = public). Written on the
        stored raw file and pages (as :func:`~scinr.newton.utils.tenancy.tenant_key`)
        and stamped on the ``IntermediateDocument``.
    created_by_user_id, job_id:
        Provenance of the upload, written on the stored raw file and pages and
        stamped on the ``IntermediateDocument``.

    Returns
    -------
    tuple[list[tuple[Path, Path | None, IntermediateDocument]], list[tuple[Path, str]]]
        ``(written, failures)`` where ``written`` is the list of successfully
        converted ``(raw_source, json_written, intermediate_document)``
        triples contributed by this entry (``json_written`` is ``None`` when
        *output_dir* is ``None``), and ``failures`` is the list of
        ``(entry_path, error_message)`` pairs for file-level conversion
        errors directly attributable to this entry.

        When *entry* is a subdirectory, any failures from the recursive
        conversion are logged there but intentionally **not** re-surfaced in
        the returned ``failures`` list (an empty list is always returned in
        that case) — this mirrors the pre-existing behaviour of
        ``convert_folder``, where only errors from files listed directly at
        a given directory level were added to that level's error tally.
    """
    if entry.is_dir():
        if output_dir is None:
            raise ValueError(f"output_dir is required to convert a directory entry: {entry}")
        # Recurse into subdirectory. See "Design decision" above: a fresh
        # semaphore is created inside the recursive convert_folder() call
        # rather than sharing this level's semaphore instance.
        sub_prefix = (_relative_prefix / entry.name) if _relative_prefix else Path(entry.name)
        sub_written, _sub_failures = await convert_folder(
            entry,
            output_dir,
            dry_run=dry_run,
            raw_file_repo=raw_file_repo,
            page_repo=page_repo,
            _relative_prefix=sub_prefix,
            context_instructions=context_instructions,
            parallel_docs=parallel_docs,
            tenant_id=tenant_id,
            created_by_user_id=created_by_user_id,
            job_id=job_id,
        )
        return sub_written, []

    if not entry.is_file():
        return [], []

    try:
        converter = get_converter(entry)
    except UnsupportedFormatError:
        logger.warning("Skipping %s: unsupported format", entry.name)
        return [], []

    # Determine relative folder path (None for files at root level of original input_dir)
    folder_path_str: str | None = str(_relative_prefix) if _relative_prefix else None

    # Output path: mirror subdir structure inside output_dir (None = in-memory only)
    file_output_dir: Path | None = None
    output_path: Path | None = None
    if output_dir is not None:
        file_output_dir = output_dir / _relative_prefix if _relative_prefix else output_dir
        output_path = file_output_dir / f"{entry.stem}.json"

    if dry_run:
        logger.info(
            "DRY-RUN: would convert %s → %s (folder_path=%s)",
            entry,
            output_path,
            folder_path_str,
        )
        return [], []

    try:
        # 1. Store original file in MongoDB (if repo provided); the repo reads
        #    it itself (streaming where supported), never materialised here.
        raw_file_id = None
        if raw_file_repo is not None:
            raw_file_id = await raw_file_repo.store_file(
                path=entry,
                filename=entry.name,
                content_type=_guess_content_type(entry),
                folder_path=folder_path_str,
                tenant_id=tenant_id,
                created_by_user_id=created_by_user_id,
                job_id=job_id,
            )

        # 2. Convert to IntermediateDocument (existing behaviour)
        doc = await _run_convert(converter, entry)
        # Inject metadata
        doc.folder_path = folder_path_str
        doc.raw_file_id = raw_file_id  # None when no repo
        doc.context_instructions = context_instructions
        doc.document_name = entry.stem  # stem of the original source file
        doc.tenant_id = tenant_id
        doc.created_by_user_id = created_by_user_id
        doc.job_id = job_id

        # 3. Store converted pages in MongoDB (if repos provided)
        if page_repo is not None and raw_file_id is not None:
            for page in doc.pages:
                page.page_id = await page_repo.store_page(
                    raw_file_id=raw_file_id,
                    filename=entry.stem,
                    folder_path=folder_path_str,
                    page_index=page.index,
                    markdown=page.markdown,
                    tenant_id=tenant_id,
                    created_by_user_id=created_by_user_id,
                    job_id=job_id,
                )

        # 4. Write JSON to output (now includes page_ids and raw_file_id),
        #    unless the caller asked for an in-memory-only result.
        if output_path is not None and file_output_dir is not None:
            file_output_dir.mkdir(parents=True, exist_ok=True)
            output_path.write_text(doc.to_json(), encoding="utf-8")
            logger.info(
                "Written: %s (%d page(s), folder_path=%s)",
                output_path,
                len(doc.pages),
                folder_path_str,
            )
        return [(entry, output_path, doc)], []
    except ConversionError as exc:
        logger.error("Conversion error for %s: %s", entry.name, exc)
        return [], [(entry, str(exc))]
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error converting %s: %s", entry.name, exc)
        return [], [(entry, str(exc))]


async def convert_folder(
    input_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    raw_file_repo: RawFileRepository | None = None,
    page_repo: PageRepository | None = None,
    _relative_prefix: Path | None = None,  # internal: relative path from the original input_dir
    context_instructions: str | None = None,
    parallel_docs: int = 1,
    *,
    tenant_id: str | None = None,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
) -> tuple[list[tuple[Path, Path, IntermediateDocument]], list[tuple[Path, str]]]:
    """Convert all supported files in *input_dir* (recursively) to *output_dir*.

    Files at the root of *input_dir* are converted with folder_path=None.
    Files inside subdirectories get folder_path set to their relative parent path
    from the original input_dir root (e.g. "ModuloA/SubModulo").

    The output directory structure mirrors the input structure::

        input_dir/ModuloA/SubModulo/doc.pdf → output_dir/ModuloA/SubModulo/doc.json

    Entries at the current directory level are converted concurrently via
    :func:`convert_one`, bounded by a fresh ``asyncio.Semaphore(parallel_docs)``
    created for *this* call. Recursive calls (for subdirectory entries) each
    create their own independent semaphore of the same size — see the
    "Design decision" note in :func:`convert_one` for why the budget is
    per-directory-level rather than shared globally across the recursion
    tree (in short: sharing a single semaphore instance would self-deadlock
    whenever a directory entry holds the semaphore for its whole nested
    conversion). With the default ``parallel_docs=1``, every level runs
    strictly one entry at a time, so the whole tree is walked exactly as
    before — this makes the observable behaviour identical to the previous
    sequential ``for`` loop for the required default case.

    Errors are isolated per entry (soft-abort): a failure converting one
    entry does not prevent the others from being converted.

    Parameters
    ----------
    input_dir:
        Directory containing source files to convert.
    output_dir:
        Directory where converted JSON files will be written.
    dry_run:
        If ``True``, logs what would be done but writes nothing.
    raw_file_repo:
        Optional :class:`~storage.base.RawFileRepository` for persisting the
        original binary file.  When ``None``, raw storage is skipped.
    page_repo:
        Optional :class:`~storage.base.PageRepository` for persisting
        converted pages.  When ``None``, page storage is skipped.
    _relative_prefix:
        Internal parameter for recursive calls. Tracks the relative path
        of input_dir from the original root. Do not pass this manually.
    context_instructions:
        Optional free-text context injected into each ``IntermediateDocument``
        before it is written to disk.
    parallel_docs:
        Maximum number of entries converted concurrently within this
        directory level (and, independently, within each subdirectory level
        during recursion). Defaults to ``1`` (sequential, matching
        pre-existing behaviour). Must be ``>= 1``.
    tenant_id:
        Tenant owning the converted file(s) (``None`` = public). Written on the
        stored raw file and pages (as :func:`~scinr.newton.utils.tenancy.tenant_key`)
        and stamped on the ``IntermediateDocument``.
    created_by_user_id, job_id:
        Provenance of the upload, written on the stored raw file and pages and
        stamped on the ``IntermediateDocument``.

    Returns
    -------
    tuple[list[tuple[Path, Path, IntermediateDocument]], list[tuple[Path, str]]]
        ``(written, failures)`` where ``written`` is the list of
        ``(raw_source, json_written, intermediate_document)`` triples for
        every file successfully converted, and ``failures`` is the list of
        ``(entry_path, error_message)`` pairs for every file-level
        conversion error directly attributable to this directory level.
        ``written`` is always empty when *dry_run* is ``True``.

    Raises
    ------
    ValueError
        If *parallel_docs* is less than ``1`` (an ``asyncio.Semaphore(0)`` or
        negative-sized semaphore can never be acquired, which would hang the
        conversion indefinitely instead of failing clearly).
    """
    if parallel_docs < 1:
        # Guard against asyncio.Semaphore(0) (or negative), which can never
        # be acquired and would hang the whole conversion indefinitely with
        # no clear error. Re-validated on every recursive call (see
        # docstring above) — harmless since the same already-validated
        # value is forwarded unchanged at each recursion level.
        raise ValueError(f"parallel_docs must be >= 1, got {parallel_docs}.")

    written: list[tuple[Path, Path, IntermediateDocument]] = []
    failures: list[tuple[Path, str]] = []

    # Fresh semaphore per convert_folder() call/recursion level — see the
    # "Design decision" note in convert_one()'s docstring for why this is
    # not shared across recursion depths.
    semaphore = asyncio.Semaphore(parallel_docs)

    entries = sorted(input_dir.iterdir())

    async def _convert_one_bounded(
        entry: Path,
    ) -> tuple[list[tuple[Path, Path, IntermediateDocument]], list[tuple[Path, str]]]:
        async with semaphore:
            return await convert_one(
                entry,
                output_dir,
                dry_run=dry_run,
                raw_file_repo=raw_file_repo,
                page_repo=page_repo,
                _relative_prefix=_relative_prefix,
                context_instructions=context_instructions,
                parallel_docs=parallel_docs,
                tenant_id=tenant_id,
                created_by_user_id=created_by_user_id,
                job_id=job_id,
            )

    tasks = [asyncio.create_task(_convert_one_bounded(entry)) for entry in entries]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for entry, result in zip(entries, results, strict=True):
        if isinstance(result, BaseException):
            # Soft-abort per entry: an unexpected failure while scheduling or
            # running one entry's task must not cancel or block the
            # conversion of the remaining entries.
            logger.error("Unexpected error processing %s: %s", entry, result)
            failures.append((entry, str(result)))
            continue
        entry_written, entry_failures = result
        written.extend(entry_written)
        failures.extend(entry_failures)

    if _relative_prefix is None:
        # Top-level call: log summary
        total_count = len(written) + len(failures)
        logger.info(
            "Folder conversion: %d converted, %d errors",
            len(written),
            len(failures),
        )
        if failures:
            logger.warning(
                "%d file(s) failed to convert out of %d total. Check logs above for details.",
                len(failures),
                total_count,
            )

    return written, failures


async def convert_single_file(
    path: Path,
    output_dir: Path,
    dry_run: bool = False,
    raw_file_repo: RawFileRepository | None = None,
    page_repo: PageRepository | None = None,
    context_instructions: str | None = None,
    *,
    tenant_id: str | None = None,
    created_by_user_id: str | None = None,
    job_id: str | None = None,
) -> Path | None:
    """Convert a single file to *output_dir*.

    Parameters
    ----------
    path:
        Source file to convert.
    output_dir:
        Directory where the converted JSON file will be written.
    dry_run:
        If ``True``, logs what would be done and returns ``None``.
    raw_file_repo:
        Optional :class:`~storage.base.RawFileRepository` for persisting the
        original binary file.  When ``None``, raw storage is skipped.
    page_repo:
        Optional :class:`~storage.base.PageRepository` for persisting
        converted pages.  When ``None``, page storage is skipped.
    context_instructions:
        Optional free-text context injected into the ``IntermediateDocument``
        before it is written to disk.
    tenant_id:
        Tenant owning the converted file(s) (``None`` = public). Written on the
        stored raw file and pages (as :func:`~scinr.newton.utils.tenancy.tenant_key`)
        and stamped on the ``IntermediateDocument``.
    created_by_user_id, job_id:
        Provenance of the upload, written on the stored raw file and pages and
        stamped on the ``IntermediateDocument``.

    Returns
    -------
    Path | None
        Path of the written file, or ``None`` when *dry_run* is ``True``.

    Raises
    ------
    UnsupportedFormatError
        If no converter is registered for the file's extension.
    ConversionError
        If the conversion fails.
    FileNotFoundError
        If *path* does not exist.
    """
    converter = get_converter(path)
    output_path = output_dir / f"{path.stem}.json"

    if dry_run:
        logger.info("DRY-RUN: would convert %s → %s", path, output_path)
        return None

    if raw_file_repo is None and page_repo is None:
        # No storage: convert via _run_convert() and write the JSON
        # directly, instead of delegating to convert_and_write() (base.py).
        # Delegating would silently break for is_async=True converters
        # (e.g. PdfConverter): convert_and_write() calls self.convert(source)
        # as a plain synchronous call, which for an async convert() just
        # builds an un-awaited coroutine object instead of running it, and
        # the subsequent doc.to_json() call then fails with a confusing
        # AttributeError far removed from the real cause (base.py's
        # convert_and_write() now also guards against direct misuse of this
        # kind — see its ConversionError check — but this path avoids it
        # entirely by not calling it). Rebuilding the same steps here
        # (output dir creation, collision-safe output path via
        # _resolve_output_path(), JSON write) keeps this path correct for
        # both sync and async converters, and additionally closes a
        # pre-existing gap: context_instructions is now injected on this
        # path too (previously only the storage-aware branch below did).
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_output_path = converter._resolve_output_path(path, output_dir)
        try:
            doc = await _run_convert(converter, path)
        except ConversionError:
            raise
        except Exception as exc:
            raise ConversionError(f"Failed to convert {path}: {exc}") from exc
        doc.context_instructions = context_instructions
        doc.document_name = path.stem
        doc.tenant_id = tenant_id
        doc.created_by_user_id = created_by_user_id
        doc.job_id = job_id
        resolved_output_path.write_text(doc.to_json(), encoding="utf-8")
        logger.info("Written: %s (%d page(s))", resolved_output_path, len(doc.pages))
        return resolved_output_path

    # With storage: full process with MongoDB
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Store original file (streamed by the repo where supported)
    raw_file_id = await raw_file_repo.store_file(
        path=path,
        filename=path.name,
        content_type=_guess_content_type(path),
        folder_path=None,
        tenant_id=tenant_id,
        created_by_user_id=created_by_user_id,
        job_id=job_id,
    )

    # 2. Convert to IntermediateDocument
    doc = await _run_convert(converter, path)
    doc.raw_file_id = raw_file_id
    doc.context_instructions = context_instructions
    doc.document_name = path.stem
    doc.tenant_id = tenant_id
    doc.created_by_user_id = created_by_user_id
    doc.job_id = job_id

    # 3. Store converted pages
    if page_repo is not None:
        for page in doc.pages:
            page.page_id = await page_repo.store_page(
                raw_file_id=raw_file_id,
                filename=path.stem,
                folder_path=None,
                page_index=page.index,
                markdown=page.markdown,
                tenant_id=tenant_id,
                created_by_user_id=created_by_user_id,
                job_id=job_id,
            )

    # 4. Write JSON (now includes raw_file_id and page_ids)
    output_path.write_text(doc.to_json(), encoding="utf-8")
    logger.info("Written: %s (%d page(s))", output_path, len(doc.pages))
    return output_path


def convert_api(
    config_path: Path,
    url: str,
    output_dir: Path,
    headers: dict[str, str],
    api_type: str = "json",
    dry_run: bool = False,
) -> Path | None:
    """Convert an API response to the intermediate document format.

    Dispatches to :class:`~converters.api_json.ApiJsonConverter` or
    :class:`~converters.api_xml.ApiXmlConverter` based on *api_type*.
    The output file is named ``{document_name}.json`` where
    ``document_name`` is read from the config file.

    Parameters
    ----------
    config_path:
        Path to the YAML or JSON converter config file.
    url:
        URL of the API endpoint to fetch.
    output_dir:
        Directory where the converted JSON file will be written.
    headers:
        HTTP headers to include in API requests.
    api_type:
        API response format: ``"json"`` (default) or ``"xml"``.
    dry_run:
        If ``True``, logs what would be done and returns ``None``.

    Returns
    -------
    Path | None
        Path of the written file, or ``None`` when *dry_run* is ``True``.

    Raises
    ------
    ConversionError
        If the API request or conversion fails.
    KeyError
        If ``document_name`` is absent from the config file.
    ValueError
        If *api_type* is not ``"json"`` or ``"xml"``.
    """
    document_name = _load_document_name(config_path)
    output_path = output_dir / f"{document_name}.json"

    if dry_run:
        logger.info("DRY-RUN: would fetch %s (%s) → %s", redact_uri(url), api_type, output_path)
        return None

    if api_type == "json":
        converter = ApiJsonConverter.from_config_file(config_path, headers)
        document = converter.convert_from_url(url)
    elif api_type == "xml":
        converter = ApiXmlConverter.from_config_file(config_path, headers)
        document = converter.convert_from_url(url)
    else:
        raise ValueError(f"Unknown api_type {api_type!r}. Expected 'json' or 'xml'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document.to_json(), encoding="utf-8")
    logger.info(
        "Written API document '%s' → %s (%d page(s))",
        document_name,
        output_path,
        len(document.pages),
    )
    return output_path
