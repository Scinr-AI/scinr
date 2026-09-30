"""Run with: .venv/bin/python -m unittest discover -s tests/scripts."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HOOK = Path(__file__).resolve().parents[2] / "scripts/build_llms_txt.py"


class LlmsBuildTest(unittest.TestCase):
    def test_every_build_syncs_current_public_docs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docs = root / "docs"
            docs.mkdir()
            (docs / "index.md").write_text("# Example\n\n> Current summary\n\n---\n")
            (docs / "guide.md").write_text("# Guide\n\nOriginal content\n")
            (docs / "private.md").write_text("# Private\n\nExcluded content\n")
            config = root / "mkdocs.yml"
            config.write_text(
                f"site_name: Example\nhooks:\n  - {HOOK}\n"
                "exclude_docs: private.md\nnav:\n  - Overview: index.md\n"
                "  - Custom Guide Title: guide.md\n"
            )

            def build():
                subprocess.run(
                    [sys.executable, "-m", "mkdocs", "build", "--strict", "-f", str(config)],
                    check=True, capture_output=True, text=True,
                )
                site = root / "site"
                index = (site / "llms.txt").read_text()
                full = (site / "llms-full.txt").read_text()
                for source in ("index.md", "guide.md", "new.md"):
                    if (docs / source).exists():
                        self.assertIn(f"]({source})", index)
                        self.assertIn((docs / source).read_text(), full)
                        self.assertEqual((site / source).read_text(), (docs / source).read_text())
                self.assertNotIn("private.md", index + full)
                self.assertFalse((site / "private.md").exists())
                return index, full

            index, _ = build()
            self.assertIn("> Current summary", index)
            self.assertIn("[Custom Guide Title](guide.md)", index)
            (docs / "index.md").write_text("# Example\n\n> Updated summary\n\n---\n")
            (docs / "guide.md").write_text("# Guide\n\nUpdated content\n")
            (docs / "new.md").write_text("# New\n\nNew page\n")
            config.write_text(config.read_text() + "  - New page: new.md\n")
            index, full = build()
            self.assertIn("> Updated summary", index)
            self.assertNotIn("Original content", full)
            self.assertNotIn("Current summary", index)


if __name__ == "__main__":
    unittest.main()
