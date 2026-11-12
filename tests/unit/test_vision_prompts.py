"""WP5 (0.5.0): the user-prompt context depends on the vision mode (D6)."""

from __future__ import annotations

from scinr.newton.converters.vision_prompts import system_prompt, user_prompt


def test_frame_mode_presents_the_context_as_the_segment_transcript():
    prompt = user_prompt("frame", context="habla desde 0")
    assert (
        "\n\nTranscript of the video segment this frame belongs to (context only, "
        "do not repeat it; use it to identify people, topics and what is on screen):\n"
        "<transcript>\nhabla desde 0\n</transcript>"
    ) in prompt
    assert "<page>" not in prompt


def test_frame_system_prompt_focuses_on_what_is_seen():
    assert "what is seen, never summarize what is said" in system_prompt("frame")


def test_embedded_mode_does_not_change():
    assert user_prompt("embedded") == "Describe this image from the document."
    assert user_prompt("embedded", context="Texto de la página") == (
        "Describe this image from the document."
        "\n\nText of the page where the image appears (context only, do not repeat it):\n"
        "<page>\nTexto de la página\n</page>"
    )
