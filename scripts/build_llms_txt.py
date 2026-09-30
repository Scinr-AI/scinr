#!/usr/bin/env python3
"""Generate LLM documentation from the same sources and config as the MkDocs site."""

from pathlib import Path

from mkdocs.config import load_config
from mkdocs.structure.files import get_files
from mkdocs.structure.nav import get_navigation


def on_post_build(config, **kwargs) -> None:
    site_dir = Path(config.site_dir)
    files = get_files(config)
    nav = get_navigation(files, config)
    pages = [page for page in nav.pages if page.file.inclusion.is_included()]
    overview = (Path(config.docs_dir) / "index.md").read_text(encoding="utf-8")
    # Use the homepage introduction so product descriptions cannot drift independently.
    index_chunks = [overview.split("\n---", 1)[0].rstrip(), "\n\n## Documentation\n"]
    full_chunks = ["# scinr Full Documentation\n\n---\n"]

    for page in pages:
        rel_path = page.file.src_uri
        content = Path(page.file.abs_src_path).read_text(encoding="utf-8")
        index_chunks.append(f"- [{page.title}]({rel_path})\n")
        full_chunks.append(f"## File: {rel_path}\n\n{content}\n\n---\n")
        target = site_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    index_chunks.append("\n## Full Documentation\n\n- [All documentation](llms-full.txt)\n")
    (site_dir / "llms.txt").write_text("".join(index_chunks), encoding="utf-8")
    (site_dir / "llms-full.txt").write_text("".join(full_chunks), encoding="utf-8")


if __name__ == "__main__":
    on_post_build(load_config())
