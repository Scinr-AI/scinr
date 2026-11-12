# Multimodal Ingestion

Since 0.5.0 `scinr` ingests images, audio and video, and describes the images
embedded in PDF, DOCX and PPTX. Everything is turned into **text first**:
images become descriptions, audio becomes a transcription, and video becomes both.
From there the document follows the usual pipeline (structure extraction,
ingestion, annotation, entity extraction) with no change. The prompts, the graph
model, deletion, freezing and progress events are unchanged.

| Input | What it becomes | Needs |
| --- | --- | --- |
| Images embedded in `.pdf`, `.docx`, `.pptx` | A description block in the page where the image is | `vision_llm` (and `describe_embedded_images=True`, the default) |
| `.jpg` `.jpeg` `.png` `.webp` `.tif` `.tiff` `.heic` `.heif` | One page: visible text transcribed, then a description | `vision_llm`; HEIC/HEIF also need `scinr[media]` |
| `.mp3` `.m4a` `.wav` `.ogg` `.flac` | One page per segment, with its transcription | `transcriber`, `ffmpeg` |
| `.mp4` `.mov` `.mkv` `.webm` | One page per segment: transcription, then described frames | `ffmpeg`; `transcriber` if it has audio; `vision_llm` for the frames |

The image, audio and video converters are **always registered**. A folder with
such files no longer skips them: they are converted or, when a requirement is
missing, they become **failed documents** with an error that says what to
configure. Those failed documents count for `on_partial_failure`.

---

## Installation and requirements

```bash
pip install "scinr[media]"   # optional: local transcription (faster-whisper) and HEIC/HEIF images
```

- **`ffmpeg` and `ffprobe`** are system binaries, needed for audio and video.
  They are not installed with the package: `brew install ffmpeg` (macOS),
  `apt-get install ffmpeg` (Debian/Ubuntu). They are looked up in the `PATH`,
  or you can set `configure(ffmpeg_path=..., ffprobe_path=...)` (env
  `FFMPEG_PATH`, `FFPROBE_PATH`).
- **A vision model** (`vision_llm`): any LangChain `BaseChatModel` that accepts
  images. It does not need structured output.
- **A transcriber** (`transcriber`): an instance of `scinr.newton.Transcriber`.

Nothing changes if you upgrade without configuring any of this. Embedded
images are only described when a `vision_llm` is configured; without one,
nothing extra is requested from Mistral and no image blob is read.

---

## Configuring the providers

Both providers are configurable and based on open interfaces. The vision model
is a LangChain chat model. The transcriber is a small interface with a
ready-made client for the OpenAI-compatible `/audio/transcriptions` endpoint,
which cloud and self-hosted servers both implement.

### Vision

`vision_llm` **never falls back to `llm`**. Vision depends on a multimodal model
that you configure on purpose, and it can be cheaper than the extraction model.

```python
from scinr.newton import configure

# Cloud, OpenAI
from langchain_openai import ChatOpenAI
configure(llm=..., vision_llm=ChatOpenAI(model="gpt-4o-mini"))

# A self-hosted OpenAI-compatible server (vLLM, LM Studio, ...)
configure(llm=..., vision_llm=ChatOpenAI(
    model="Qwen/Qwen2.5-VL-7B-Instruct", base_url="http://gpu-server:8000/v1", api_key="none",
))

# AWS Bedrock (pip install "scinr[bedrock]")
from langchain_aws import ChatBedrockConverse
configure(llm=..., vision_llm=ChatBedrockConverse(model="eu.amazon.nova-lite-v1:0"))

# Ollama (pip install "scinr[ollama]"), local or on a shared server
from langchain_ollama import ChatOllama
configure(llm=..., vision_llm=ChatOllama(model="qwen3-vl:4b", base_url="http://gpu-server:11434"))
```

The image is sent as a standard LangChain image block
(`{"type": "image", "base64": ..., "mime_type": "image/jpeg"}`), which every
provider integration translates to its own format. Before sending, the image is
downscaled so that its longer side is at most `image_max_side_px` (1568). Calls
go through the same semaphore as the other LLM calls (`llm_concurrency`), with
the usual retries on throttling.

> **Small machines.** Running a vision model inside small workers (around
> 2 GB of RAM) is not practical. The models that read data reliably need
> 1.6–2 GB of memory for themselves, and on CPU they take tens of seconds per
> image. For bulk ingestion, point `vision_llm` (and the transcriber) at a
> shared inference server or a cloud API; the workers then only convert and
> stream, which stays bounded in memory.

### Transcription

```python
from scinr.newton.media.transcribers import (
    FasterWhisperTranscriber,
    OpenAICompatibleTranscriber,
)

# Cloud (OpenAI or any compatible provider)
configure(transcriber=OpenAICompatibleTranscriber(model="whisper-1", api_key="sk-..."))

# A self-hosted OpenAI-compatible server: no key needed
configure(transcriber=OpenAICompatibleTranscriber(
    model="Systran/faster-whisper-small", base_url="http://asr-server:8000/v1",
))

# Local inference in the same process (pip install "scinr[media]")
configure(transcriber=FasterWhisperTranscriber(model_size="small", device="auto"))
```

`OpenAICompatibleTranscriber(model, *, api_key=None, base_url="https://api.openai.com/v1", timeout=300, max_retries=3, no_speech_threshold=0.6)`:

- streams the segment file from disk in a multipart body;
- sends no `Authorization` header when `api_key` is `None`;
- retries network errors and HTTP 429 / 5xx with exponential backoff;
- asks for segment timestamps (`response_format=verbose_json`,
  `timestamp_granularities[]=segment`). Servers and models that reject those
  parameters (HTTP 400 / 422) get the segment again as `json`, and the instance
  remembers it. Without timestamps the segment is a single block of text, and
  the page times still come from the cut, so traceability is the same;
- drops the segments the model itself marks as silence: `no_speech_prob`
  above `no_speech_threshold` and `avg_logprob` below `-1`, Whisper's own
  rule. It needs `verbose_json`; `None` keeps every segment.

`FasterWhisperTranscriber(model_size="small", device="auto", compute_type="default", *, vad_filter=True, vad_parameters=None)`
loads the model once per process, caches it by those three values, and
transcribes one segment at a time per model, in a worker thread. With
`vad_filter=True` (unlike faster-whisper's own default) its bundled
voice-activity detector skips the stretches without speech before
transcribing. `vad_parameters` passes faster-whisper's `VadOptions`, for
example `{"min_silence_duration_ms": 500}`.

**Silence.** Whisper models tend to invent a sentence ("Thanks for
watching") on silent or music-only audio. The two filters above exist for
that, and they are on by default. A custom `Transcriber` should return no
segments for silence, so that the segment produces no page.

**Cloud or local.** Any provider with the OpenAI `/audio/transcriptions`
endpoint works through `OpenAICompatibleTranscriber`: OpenAI itself, or a
self-hosted server. A provider with its own API (AWS Transcribe, Google
Speech-to-Text…) takes a custom `Transcriber` (next section). Since the
transcriber is an injected object, a custom one can also wrap two others and
fall back from one to the other when the first fails.

### Writing your own `Transcriber`

Any other provider plugs in by implementing one method:

```python
from pathlib import Path

from scinr.newton import Transcriber, TranscriptSegment


class MyTranscriber(Transcriber):
    async def transcribe(self, path: Path, language: str | None = None) -> list[TranscriptSegment]:
        result = await my_client.transcribe(path.read_bytes(), language=language)
        return [
            TranscriptSegment(start_ms=int(s.start * 1000), end_ms=int(s.end * 1000), text=s.text)
            for s in result.segments
        ]


configure(transcriber=MyTranscriber())
```

The contract:

- `transcribe()` is called **once per segment** with a temporary **FLAC file,
  mono, 16 kHz**, at most `media_segment_seconds` long. The converter cuts the
  segments with `ffmpeg`, so your transcriber never sees the whole medium. The
  file is deleted right after the call returns.
- Times are **relative to the start of that file**. The converter adds the
  segment offset.
- It may return a **single segment** with all the text. An empty list (or
  empty texts) means silence, and the segment produces no page.
- `language` is an ISO 639-1 code (`media_language`), or `None` for
  auto-detection.
- To fail, **raise any exception**. The converter applies
  `media_error_strategy` to the segment: `best_effort` skips it and records it
  in `missing_segments`, `fail_fast` fails the document. Retries are your
  implementation's job.

---

## Embedded images (PDF, DOCX, PPTX)

With a `vision_llm` configured and `describe_embedded_images=True` (the
default), each embedded image is described and the description is written
into the page markdown:

```text
## 3. Results

<image index="2">
Bar chart of quarterly sales in 2025 (millions EUR):

| Quarter | Sales |
| --- | --- |
| Q1 | 12 |
| Q2 | 18 |
</image>

The second quarter...
```

- The block **replaces the image reference** `![…](ref)` in the markdown when
  the page has one (PDF via Mistral OCR, DOCX). Otherwise it goes **at the end
  of the page**, in image order (PPTX).
- **Trivial images are skipped** before anything is sent:
  - images whose shorter side is below `image_min_side_px` (64);
  - an image identical to one already seen in the document (a logo on every
    page is described once);
  - beyond `max_images_per_page` (10) per page, the largest are kept.

  A skipped image's reference is removed from the markdown.
- An empty description (the model found nothing informative) inserts nothing.
- The extraction prompts are unchanged: the description is just page text.

**PDF.** The OCR request asks Mistral for the images
(`include_image_base64`, with `image_min_size` and an `image_limit` bound).
Responses with images are much heavier than text only, so these PDFs are sent
in chunks of `mistral_ocr_image_max_pages` (50) pages. Each chunk's images are
described, and their bytes released, before the next chunk is sent.

**DOCX.** The pictures of body paragraphs are covered, inline or floating.
Images in headers, footers and table cells are not.

**PPTX.** Every picture of a slide is covered, including those inside group
shapes. With description off, the
intermediate JSON of a PPTX no longer carries the base64 of its images. Before
0.5.0 it did, and nothing read it.

No base64 of an image exists until that image is described, and it is dropped
right after. The intermediate JSON never carries image bytes: an `images`
entry keeps its `ref`, size and `description`, with an empty `base64`.

## Image files

An image file is a one-page document. The `vision_llm` first transcribes all
visible text, keeping its structure (headings, lists, tables), and then
describes the image. EXIF orientation is applied before sending. A failure is
always a `ConversionError`, because there is nothing partial to keep.

## Audio

Audio is cut into segments of `media_segment_seconds` (300 s). Each segment is
one page with its time range and a header:

```text
[00:05:00 – 00:10:00]

first transcribed segment
second transcribed segment
```

- The page times come from the cut, not from the model.
- The last segment lasts at least `media_min_segment_seconds` (10 s). When the
  tail would be shorter, the previous cut moves earlier, to a whole second:
  300.05 s in 300 s segments gives `[00:00:00 – 00:04:50]` and
  `[00:04:50 – 00:05:00]` (10.05 s). A medium of a few milliseconds would
  otherwise end in a segment that some providers reject (OpenAI asks for at
  least 0.1 s). No segment is longer than `media_segment_seconds`. `0`
  disables the adjustment.
- A silent segment produces no page. The following pages keep their index, so
  the index is always `segment number`.
- An audio longer than `max_media_duration_seconds` (4 h) fails **before**
  anything is transcribed. Pass `None` for no limit.

## Video

Same segments as audio. Each page holds the header, the transcription (when
the video has an audio track) and one block per sampled frame:

```text
[00:05:00 – 00:10:00]

transcribed speech of the segment

<image index="0" time="00:05:00">
A presenter in front of a slide titled "Q3 results"; on screen: "Revenue +12 %".
</image>

<image index="1" time="00:05:30">
...
</image>
```

- Frames are sampled uniformly at `video_frames_per_minute` (2.0, so 10 per
  5-minute segment). A frame identical to the previous one is skipped. `0`
  means transcription only. There is no scene detection in this version.
- Each frame is described in a call of its own, with one image, and gets **the
  transcript of its segment as context**, the way an embedded image gets the
  text of its page. The model uses it to name the topic or the speaker on
  screen, but describes what is seen, not what is said. Without a transcript
  (a video without audio, or a segment whose transcription failed in
  `best_effort`) the frame is described without context. Frames are not sent
  several to a call: that would raise the memory per segment, many local
  models accept a single image, and one failure would lose the whole segment.
- A video **with** audio needs a `transcriber`. A video **without** audio is
  processed from frames only.
- Without a `vision_llm` the frames are skipped with a warning, and the video
  keeps its transcription. With neither source available, the conversion fails.

## Playing a segment

Each audio or video page carries `start_ms` / `end_ms`, the range
`[start_ms, end_ms)` in the original file. It is stored with the page
(`converted_pages` in MongoDB) and exposed on `PageText`, so a search result
can be played from the stored original:

```python
pages = await nav.get_info_unit_source_text(info_unit_uid)   # list[PageText]
page = pages[0]
original = await nav.get_document_original("Courses/class-03.mp4")   # OriginalFile | None
if original is not None and page.start_ms is not None:
    print(original.record.filename, page.start_ms / 1000, page.end_ms / 1000)
    async for chunk in original.stream:          # constant memory
        player.feed(chunk)                       # then seek to page.start_ms
```

`start_ms` / `end_ms` are `None` for documents without a time axis (PDF,
DOCX…) and for pages stored before 0.5.0.

**Jumping to the exact sentence.** Each page also carries `transcript`: the
transcribed sentences with their own range, as `TranscriptCue(start_ms,
end_ms, text)`, with times absolute in the original file (not relative to the
segment). `PageText.find_in_transcript(text)` returns the cues that contain a
quote, compared without case or repeated whitespace, so a quote from the
source text leads to its second:

```python
cues = page.find_in_transcript("revenue grew twelve percent")
if cues:
    player.seek(cues[0].start_ms)
```

The cues are stored with the page (`converted_pages.transcript`), not in the
markdown: the text the extraction reads is the same as without them and costs
no extra tokens. When the transcriber returns no timestamps (an
OpenAI-compatible server that only answers `response_format=json`), each page
gets a single cue covering the whole segment; times are never invented.
`transcript` is `None` for documents without a time axis and for pages stored
before this change.

## Retrieving images and frames

Each image that gets a description is stored in the document base, linked to
its page, so it can be shown to the user as it was:

- **Embedded images** (DOCX, PPTX, PDF): the original bytes, not resized. For
  DOCX and PPTX, the picture as stored in the package. For PDF, the crop that
  Mistral OCR returns, which is what was described and what the page shows.
- **Video frames**: the JPEG that was described, already scaled to
  `image_max_side_px`, with its `time_ms`. The full-resolution frame can be
  taken from the original video at that instant.
- **Not stored**: images skipped as trivial, repeated or over
  `max_images_per_page`, images that could not be read or described, and
  images the model left without a description. A repeated image (a logo on
  every page) is stored once, at its first appearance. Image files (`.png`,
  `.jpg`…) are not stored again: their original is the uploaded file
  (`get_document_original`).

An image is identified by `(raw_file_id, page_index, image_index)`, where
`image_index` is the `index` of its `<image index="N">` block. The markdown
carries no image id.

```python
pages = await nav.get_document_source_text("report.docx", include_images=True)
for page in pages:
    for ref in page.images:                      # list[PageImageRef]
        print(page.index, ref.image_index, ref.kind, ref.description)
        image = await nav.get_page_image(ref.image_id)   # PageImageFile | None
        if image is not None:
            async for chunk in image.stream:     # constant memory
                response.write(chunk)
```

- `include_images=True` works on `get_document_source_text`,
  `get_info_unit_source_text` and `get_structure_nodes_source_pages`.
  Without it, `PageText.images` is `None` and the image storage is not read;
  with it, a page without images gets `[]`.
- `get_document_images(document, page_indexes=..., kind="embedded" | "frame")`
  lists a document's images without reading its pages.
- `get_page_image(image_id)` opens one image by id, the natural way to serve
  `/images/{image_id}` from an API. There is no graph node to resolve from an
  id, so the storage read applies the call's **full scope** (tenant,
  `include_public`, user and job). An image outside it returns `None`, like a
  missing one. See
  [Graph Navigation](graph-navigation.md#reading-source-text).

Storing is controlled by two switches, both on by default:
`store_described_images` (`STORE_DESCRIBED_IMAGES`) for embedded images and
`store_video_frames` (`STORE_VIDEO_FRAMES`) for frames. They only act with a
persistent storage backend that stores images (MongoDB does). A custom backend
that does not implement the image methods keeps working, stores no images and
logs one warning per process (see
[Storage Backends](storage-backends.md#repository-interfaces)).

**A failed image write fails the document**, whatever `media_error_strategy`,
the same as a failed page write: a described image is never kept without its
original. The images already stored for that upload are deleted first.
`delete_document()` deletes a document's images with its pages
(`DeletionResult.page_images_deleted`).

## File metadata

Images, audio and video reach the `:Document` node with `file_*` properties.
The full list is in the
[Neo4j graph model](neo4j-graph.md#document); it covers duration, container
and codecs, width and height, frame rate, capture time and device. The
navigation API returns them in `DocumentRef.raw`, and freeze snapshots keep
them.

**Location.** With `strip_location=True` (the default) the GPS of images is not
even read, and `file_gps_latitude` / `file_gps_longitude` stay `null`. Set it to
`False` to keep them.

## Errors and partial results

`media_error_strategy` (default `best_effort`) decides what happens when one
image, segment or frame fails:

| Failure | `best_effort` | `fail_fast` |
| --- | --- | --- |
| An embedded image | Skipped; `(page_index, image_index)` added to `missing_images` | The document fails |
| An audio / video segment's transcription | Skipped; `[start_ms, end_ms)` added to `missing_segments` | The document fails, with the range in the message |
| A video frame | Skipped; `(page_index, frame_index)` added to `missing_images` | The document fails |
| An image file | Always a failed document | Always a failed document |

When ffmpeg cannot extract the frames of a video segment, every frame planned
for that segment is recorded in `missing_images` (or, with `fail_fast`, the
document fails). The segment keeps its transcription.

**Nothing left means failure.** When every part failed and nothing with text is
left, the document fails, whatever the strategy. A typical cause is a provider
that rejects every request, such as a wrong API key. The error lists what was
skipped. This applies to every converter: a PDF whose every chunk failed with
`mistral_ocr_error_strategy="best_effort"` fails in the same way. A document
that is merely empty, with no failure (a silent audio, a blank page), is not
an error.

**Where the omissions are reported.** `missing_images` and `missing_segments`
are written on the intermediate document (and its JSON), next to the PDF's
`missing_page_ranges`. The same three lists are on the document's preprocess
`DocumentResult`. A partial document counts as processed, not failed, so
`PipelineResult.success` does not change. At the end of the run:

```python
result = await run_pipeline(input_raw="inputs/")
for doc in result.documents_with_omissions:
    print(doc.document_name, doc.missing_page_ranges, doc.missing_segments, doc.missing_images)
```

`run_pipeline` also logs one closing warning per partial document. A missing
requirement (`vision_llm`, `transcriber`, `ffmpeg`, `pillow-heif` for HEIC) is
not a partial result: the document fails at Stage 0 with the usual
`stage_failed` / `document_finished` progress events.

## Memory

Peak memory depends on the size of **one unit**: one image, one segment, or one
PDF chunk. It never depends on the length of the medium or on the number of
images.

- Images are described one page at a time. Inside a page, and across
  documents, images are read, decoded and downscaled `image_decode_concurrency`
  at a time per event loop, **one** by default. Only the downscaled JPEGs wait for
  the model, with at most `llm_concurrency` calls in flight. With a real model
  this does not slow anything down, because decoding overlaps the calls (see
  [Performance Tuning](performance-tuning.md#multimodal-ingestion)). Decoding is what costs memory: a 6 MP PNG
  is about 18 MB of pixels whatever its file size. Measured with 8 PNG of
  3000×2000 on one page, the peak RSS went from 632 MiB when they were all
  decoded at once to 65 MiB. The largest single decode is bounded by Pillow's
  `Image.MAX_IMAGE_PIXELS` (about 89 MP); a larger image fails as a
  decompression bomb. JPEGs are decoded already reduced, so only lossless
  formats pay their full size.
- A PDF holds one chunk's response at a time. The response is parsed page by
  page, and its images are released before the next chunk.
- Audio and video are never read in Python. `ffmpeg` writes one temporary
  segment (or the frames of one segment) at a time.
- Storing the described images does not change the peak. The original of an
  embedded image is read again inside the same decoding lane, so no more than
  `image_decode_concurrency` originals are in memory at once. A frame is
  uploaded from disk as a stream. Measured: +0.2 MiB for a DOCX with 100
  images and for a 300-page PDF, +0.5 MiB for a video.
- At most `media_concurrency` (1) audio/video files are converted at once per
  event loop, and at most `mistral_ocr_concurrency` (2) PDFs, whatever
  `parallel_docs` is.

`scripts/memory_bench/bench_media.py` measures every case. A 2-hour audio peaks
like a 30-minute one, and a 363 MB video like a 7.8 MB one. See
[Performance Tuning](performance-tuning.md#multimodal-ingestion).

## Limitations

- No speaker diarisation (`TranscriptSegment.speaker` is reserved), no scene
  detection, and no identification of people by their appearance (the prompts
  forbid it).
- DOCX: no images from headers, footers or table cells.
- Native charts of DOCX and PPTX (inserted as a Word or PowerPoint chart, not
  pasted as a picture) are not images, so they are not described. Their data is
  not converted either. A chart pasted as a picture is described like any
  other image.
- No embeddings and no retrieval by image similarity: images are searchable
  through their text.
- A stored PDF image is the crop Mistral OCR returns, not the object embedded
  in the PDF. A stored video frame is at `image_max_side_px`, not at the
  video's resolution.
- Stored frames take space: about `video_frames_per_minute × minutes` JPEGs
  per video. Set `store_video_frames=False` to keep only the embedded images.
- Documents ingested before 0.5.0 are not migrated (see the next section).

## Upgrading: names keep the extension

Since 0.5.0 a document's `name` and `doc_path` keep the source file extension:
`Modulo/informe.pdf` is `doc_path = "Modulo/informe.pdf"` (it used to be
`"Modulo/informe"`). This is what lets `informe.pdf` and `informe.mp3` coexist
in the same folder.

- Re-ingesting a file that was already ingested creates a **new** `:Document`
  (`informe.pdf`), not a new version of the old one (`informe`).
- `replaces=`, `document_names=`, deletion, freezing and navigation take
  document names and paths: use the extension for documents ingested from 0.5.0
  on, and the old extension-less name for the older ones.
- Intermediate and extraction JSON files are written as `informe.pdf.json` and
  `extract-informe.pdf.json`. JSON files written by earlier versions are still
  ingestible and keep their extension-less name.
