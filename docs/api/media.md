# Media API

Speech-to-text and `ffmpeg` helpers used by the audio and video converters, and
the file metadata that reaches the `:Document` node. For the end-to-end
behaviour see the [Multimodal Ingestion guide](../user-guides/multimodal-ingestion.md).

`Transcriber` and `TranscriptSegment` are also exported from `scinr.newton`.

## Transcriber Interface

::: scinr.newton.media.transcriber

## Ready-made Transcribers

::: scinr.newton.media.transcribers.openai_compatible

::: scinr.newton.media.transcribers.faster_whisper

## ffmpeg Helpers

::: scinr.newton.media.ffmpeg

## File Metadata

::: scinr.newton.models.file_metadata
