# Converters API

Document format converters used in Stage 0 (Preprocess).

## Converter Registry

::: scinr.newton.converters.registry

## Base Classes

::: scinr.newton.converters.base

## Format Converters

::: scinr.newton.converters.pdf

::: scinr.newton.converters.docx

::: scinr.newton.converters.pptx

::: scinr.newton.converters.xlsx

::: scinr.newton.converters.csv

::: scinr.newton.converters.html

::: scinr.newton.converters.api_json

::: scinr.newton.converters.api_xml

::: scinr.newton.converters.text

## Image, Audio and Video Converters

Always registered; a missing requirement (`vision_llm`, `transcriber`,
`ffmpeg`) fails that document at conversion with a message that says how to
configure it. See the [Multimodal Ingestion guide](../user-guides/multimodal-ingestion.md).

::: scinr.newton.converters.image

::: scinr.newton.converters.audio

::: scinr.newton.converters.video

::: scinr.newton.converters.media_common

## Vision and Embedded-Image Enrichment

Descriptions of embedded images (PDF, DOCX, PPTX), standalone images and video
frames with the configured `vision_llm`.

::: scinr.newton.converters.vision

::: scinr.newton.converters.vision_prompts

::: scinr.newton.converters.enrichment
