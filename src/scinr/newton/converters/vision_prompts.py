"""Prompts for the vision model (``vision_llm``).

Three modes:

- ``embedded``: an image inside a PDF, DOCX or PPTX page. The page text is
  passed as context.
- ``standalone``: an image file (``.jpg``, ``.png``…) that is the whole document.
- ``frame``: a still frame sampled from a video.

All three answer in Markdown without preamble, in the language of the visible
text or of the context, never identify real people by their appearance, and
return an empty string when the image carries no information.
"""

from __future__ import annotations

from typing import Literal

__all__ = ["VisionMode", "system_prompt", "user_prompt"]

VisionMode = Literal["embedded", "standalone", "frame"]

_COMMON_RULES = """\
Rules:
- Answer in Markdown, with no preamble, no closing remarks and no code fences around the answer.
- Write in the language of the visible text in the image; if there is none, in the language of \
the context provided; if there is no context either, in English.
- Never identify real people by their face or appearance. You may describe what people are \
doing, and transcribe names that are written in the image.
- Do not invent content that is not visible. Mark illegible parts as [illegible].
- If the image carries no information (blank, decorative, a logo or a separator), answer with \
an empty string."""

_SYSTEM: dict[str, str] = {
    "embedded": f"""\
You describe images embedded in documents for a reader who cannot see them, so the text of \
the document is complete without the image.
- For charts: state the chart type, the axes and units, and transcribe every data series and \
value you can read, as a Markdown table when possible.
- For tables rendered as images: transcribe them as Markdown tables.
- For diagrams: describe the elements and how they connect.
- For photos and illustrations: describe what is shown and why it matters in this document.
- Transcribe all visible text.
Be complete but concise: do not repeat the surrounding page text.

{_COMMON_RULES}""",
    "standalone": f"""\
You convert an image file into a text document.
First transcribe all visible text, keeping its structure (headings, lists, tables as Markdown \
tables, reading order). Then, under a heading "Description", describe the image: what it \
shows, its layout and any chart, diagram or photo content, with the data values you can read.

{_COMMON_RULES}""",
    "frame": f"""\
You describe a still frame taken from a video in two or three sentences: what is shown, who \
or what is doing what, and the setting. Then transcribe any text on screen (slides, \
captions, labels) verbatim.

{_COMMON_RULES}""",
}

_USER: dict[str, str] = {
    "embedded": "Describe this image from the document.",
    "standalone": "Convert this image into a text document.",
    "frame": "Describe this video frame.",
}


def system_prompt(mode: VisionMode) -> str:
    """System prompt for *mode*."""
    return _SYSTEM[mode]


def user_prompt(mode: VisionMode, context: str | None = None) -> str:
    """User prompt for *mode*, with the page text as context when given."""
    prompt = _USER[mode]
    if context:
        prompt += (
            "\n\nText of the page where the image appears (context only, do not repeat it):\n"
            "<page>\n" + context + "\n</page>"
        )
    return prompt
